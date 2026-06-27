import os
import numpy as np
import pandas as pd
from pathlib import Path
from fractions import Fraction
from scipy.signal import butter, filtfilt, resample_poly
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- CONFIGURATION ---
SF_DICT = {"fog_star": 60.0, "omnia_park": 90.0, "pd_phone": 200.0, "wearpd": 100.0, "kiel": 200.0}
DATASETS = ["fog_star", "omnia_park", "pd_phone", "wearpd", "kiel"]
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TARGET_HZ = 64
WINDOW_SEC = 5

# Adaptive Windowing Parameters
MIN_OVERLAP = 0.5
MAX_OVERLAP = 0.85
TARGET_CLASS_RATIO = 0.85

# Subject to use for the standalone preprocessing PNG figures (set to None to skip)
VIZ_SUBJECT = ("wearpd", "wpd006")

RAW_DATA_DIR = Path("posturalInstability/data/cleaned_data")
OUTPUT_DIR = Path("posturalInstability/data/preprocessed_data/")

# =========================================================================
# 1. CLEANING AND FILTERING FUNCTIONS
# =========================================================================

def apply_bandpass(group, sf, c1 = 0.5, c2 = 20):
    nyq = 0.5 * sf
    b, a = butter(4, [c1/nyq, c2/nyq], btype="band")
    for col in SENSOR_COLS:
        valid_mask = ~group[col].isna()
        if valid_mask.sum() > 30: 
            values = group.loc[valid_mask, col].values
            padlen = min(len(values) - 1, 3 * max(len(a), len(b)))
            group.loc[valid_mask, col] = filtfilt(
                b,
                a,
                values,
                padtype="odd",
                padlen=padlen,
            )
    return group

def apply_lowpass(group, sf, cutoff=20):
    nyq = 0.5 * sf
    b, a = butter(4, cutoff/nyq, btype="low")
    for col in SENSOR_COLS:
        valid_mask = ~group[col].isna()
        if valid_mask.sum() > 30: 
            values = group.loc[valid_mask, col].values
            padlen = min(len(values) - 1, 3 * max(len(a), len(b)))
            group.loc[valid_mask, col] = filtfilt(
                b,
                a,
                values,
                padtype="odd",
                padlen=padlen,
            )
    return group

def soft_trim_outliers(group, sf, z_threshold=5):
    #   print(f"  Removing outliers with Z-score > {z_threshold} (sf={sf} Hz)")
    for col in SENSOR_COLS:
        rolling = group[col].rolling(window=int(sf*2), center=True, min_periods=1)
        z_score = np.abs((group[col] - rolling.mean()) / (rolling.std() + 1e-6))
        group.loc[z_score > z_threshold, col] = np.nan
    return group

# =========================================================================
# 2. ADAPTIVE WINDOWING LOGIC
# =========================================================================

def interpolate_nans(window):
    """Interpolate NaNs with a degree-3 polynomial if they are < 5% of the window rows."""
    nan_mask = np.isnan(window)
    nan_frac = nan_mask.any(axis=1).sum() / len(window)
    if nan_frac == 0:
        return window
    if nan_frac > 0.05:
        return None
    result = window.copy()
    for ch in range(window.shape[1]):
        col = result[:, ch]
        nan_idx = np.where(np.isnan(col))[0]
        valid_idx = np.where(~np.isnan(col))[0]
        if len(valid_idx) < 4:
            return None
        coeffs = np.polyfit(valid_idx, col[valid_idx], deg=3)
        col[nan_idx] = np.polyval(coeffs, nan_idx)
    return result

def merge_stability_label(label):
    label = int(round(float(label)))
    return 3 if label >= 3 else label

def window_step(win_size, overlap):
    return max(1, int(round(win_size * (1 - overlap))))

def count_valid_windows(data, win_size, overlap):
    """Count how many valid windows (without NaN) a segment would produce."""
    step = window_step(win_size, overlap)
    if len(data) < win_size: return 0
    count = 0
    for start_idx in range(0, len(data) - win_size + 1, step):
        if not np.isnan(data[start_idx:start_idx + win_size]).any():
            count += 1
    return count

def class_overlap_from_target(base_count, target_count):
    """Compute the overlap required to reach the target number of windows."""
    if base_count <= 0: return MIN_OVERLAP
    scale = target_count / base_count
    if scale <= 1: return MIN_OVERLAP
    
    # Formula inversa: target = base * (1-min_ov) / (1-new_ov)
    overlap = 1 - (1 - MIN_OVERLAP) / scale
    return float(np.clip(overlap, MIN_OVERLAP, MAX_OVERLAP))

def resample_group(group, original_sf):
    n_target = int(len(group) * TARGET_HZ / original_sf)
    resampled = {
        "subjectID": group["subjectID"].iloc[0],
        "sessionID": group["sessionID"].iloc[0],
        "taskID": int(group["taskID"].iloc[0]),
        "dataset": group["dataset"].iloc[0]
    }
    ratio = Fraction(int(TARGET_HZ), int(round(original_sf))).limit_denominator()
    up, down = ratio.numerator, ratio.denominator
    for col in SENSOR_COLS:
        values = group[col].values
        resampled_values = resample_poly(values, up, down)
        if len(resampled_values) > n_target:
            resampled_values = resampled_values[:n_target]
        elif len(resampled_values) < n_target:
            resampled_values = np.pad(resampled_values, (0, n_target - len(resampled_values)), mode="edge")
        resampled[col] = resampled_values
    
    if "isTurn" in group.columns:
        turn_vals = group["isTurn"].values
        turn_resampled = resample_poly(turn_vals, up, down)
        if len(turn_resampled) > n_target:
            turn_resampled = turn_resampled[:n_target]
        elif len(turn_resampled) < n_target:
            turn_resampled = np.pad(turn_resampled, (0, n_target - len(turn_resampled)), mode="edge")
        resampled["isTurn"] = np.rint(turn_resampled).astype(int)
    else:
        resampled["isTurn"] = np.zeros(n_target)
    return pd.DataFrame(resampled)

def create_windows(df, min_windows=1):

    windows, labels, metadata_rows = [], [], []
    win_size = int(TARGET_HZ * WINDOW_SEC)
    group_cols = ["subjectID", "sessionID", "dataset", "taskID", "label"]

    # --- PHASE 1: Compute Base Counts ---
    print("Analyzing distribution for Adaptive Windowing...")
    base_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for (_, _, _, _, label), group in df.groupby(group_cols):
        base_counts[label] += count_valid_windows(group[SENSOR_COLS].values, win_size, MIN_OVERLAP)
    
    print(f"  Base window counts by class: {base_counts}")
    majority_count = max(base_counts.values())
    target_count = int(majority_count * TARGET_CLASS_RATIO)
    
    # Compute class-specific overlap
    overlap_by_class = {
        cls: class_overlap_from_target(base_counts[cls], target_count) 
        for cls in base_counts
    }
    print(f"  Overlap plan: { {k: round(v, 2) for k, v in overlap_by_class.items()} }")

    # --- PHASE 2: Window Generation ---
    window_id = 0
    for (sid, sessid, ds, tid, label), group in df.groupby(group_cols):
        data = group[SENSOR_COLS].values
        turns = group["isTurn"].values
        
        current_overlap = overlap_by_class[label]
        step = window_step(win_size, current_overlap)
        
        for start_idx in range(0, len(data) - win_size + 1, step):
            win_data = data[start_idx : start_idx + win_size].copy()
            if np.isnan(win_data).any():
                win_data = interpolate_nans(win_data)
                if win_data is None:
                    continue

            windows.append(win_data)
            labels.append(label)
            metadata_rows.append({
                "window_id": window_id,
                "subjectID": sid,
                "dataset": ds,
                "taskID": tid,
                "isTurn": int(turns[start_idx : start_idx + win_size].max()),
                "label": label,
                "overlap": current_overlap,
                "age": group["age"].iloc[0],
                "gender": group["gender"].iloc[0],
                "disease_duration": group["disease_duration"].iloc[0],
            })
            window_id += 1

    x, y, meta = np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32), pd.DataFrame(metadata_rows)

    # --- PHASE 3: Quality Filter for Subjects ---
    # Garantisce che ogni soggetto abbia almeno min_windows per task (Stance OR Walk)
    meta['ds_sid'] = list(zip(meta['dataset'], meta['subjectID']))
    
    # Conta quante finestre ha ogni soggetto per ogni task
    counts = meta.groupby(['ds_sid', 'taskID']).size().unstack(fill_value=0)
    
    # Condizione: Walk (2) >= min E Stance (0 o 1) >= min
    # Usiamo .get() per gestire il caso in cui un task manchi totalmente incounts
    valid_mask = (counts.get(2, 0) >= min_windows) & \
                 ((counts.get(0, 0) >= min_windows) | (counts.get(1, 0) >= min_windows))
    
    valid_ds_sids = counts[valid_mask].index
    
    # Applica il filtro finale
    final_indices = meta[meta['ds_sid'].isin(valid_ds_sids)].index
    
    print(f"Scartate {len(x) - len(final_indices)} finestre appartenenti a soggetti incompleti o con dati insufficienti.")
    
    return x[final_indices], y[final_indices], meta.loc[final_indices].drop(columns=['ds_sid'])
# =========================================================================
# 3. VISUALIZATION
# =========================================================================

def plot_preprocessing_comparison(viz_collection, output_dir):
    """One PDF page per subject: 2 rows (Acc/Gyro) × 4 cols (Walk Before/After, Stance Before/After)."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.titleweight":  "bold",
        "axes.labelsize":    15,
        "xtick.labelsize":   13,
        "ytick.labelsize":   13,
        "legend.fontsize":   15,
        "legend.framealpha": 1,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })

    CHAN_COLORS = {"x": "#56B4E9", "y": "#E69F00", "z": "#009E73"}
    SENSOR_GROUPS = [
        (["acc_x",  "acc_y",  "acc_z"],  "Accelerometer", "g"),
        (["gyro_x", "gyro_y", "gyro_z"], "Gyroscope",     "rad/s"),
    ]
    COL_LABELS = ["Walk — Before", "Walk — After", "Stance — Before", "Stance — After"]

    complete = [(k, v) for k, v in sorted(viz_collection.items())
                if "task" in v and "stance" in v]
    

    pdf_path = output_dir / "preprocessing_all_subjects.pdf"
    with PdfPages(pdf_path) as pdf:
        for (ds_name, sid), entry in complete:
            data_seq = [
                (entry["task"]["raw"],    entry["task"]["sf"]),
                (entry["task"]["proc"],   TARGET_HZ),
                (entry["stance"]["raw"],  entry["stance"]["sf"]),
                (entry["stance"]["proc"], TARGET_HZ),
            ]

            # sharey='row': same y-scale across all 4 columns within each sensor row
            fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharey="row")
            fig.suptitle(f"Dataset: {ds_name}  |  Subject: {sid}", fontweight="bold")

            legend_handles, legend_labels = [], []

            for col_idx, (df, rate) in enumerate(data_seq):
                t = np.arange(len(df)) / rate
                for row_idx, (cols, sensor_label, unit) in enumerate(SENSOR_GROUPS):
                    ax = axes[row_idx, col_idx]
                    for col in cols:
                        axis_name = col.split("_")[1]
                        line, = ax.plot(t, df[col].values, color=CHAN_COLORS[axis_name],
                                        linewidth=0.7, label=axis_name)
                        if col_idx == 0 and row_idx == 0:
                            legend_handles.append(line)
                            legend_labels.append(axis_name)

                    if row_idx == 0:
                        ax.set_title(COL_LABELS[col_idx])
                    if col_idx == 0:
                        ax.set_ylabel(f"{sensor_label}\n({unit})", fontweight="bold")
                    if row_idx == 1:
                        ax.set_xlabel("Time (s)")

            fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3,
                       bbox_to_anchor=(0.5, -0.02), framealpha=0.85,
                       title="Axis", title_fontsize=10)
            plt.tight_layout(rect=[0, 0.06, 1, 1])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  PDF with {len(complete)} subjects saved to {pdf_path}")
    

def save_subject_pngs(entry, output_dir):
    """Save preprocessing_walk.png and preprocessing_stance.png for a single subject."""
    CHAN_COLORS = {"x": "#56B4E9", "y": "#E69F00", "z": "#009E73"}
    SENSOR_GROUPS = [
        (["acc_x",  "acc_y",  "acc_z"],  "Accelerometer", "g"),
        (["gyro_x", "gyro_y", "gyro_z"], "Gyroscope",     "rad/s"),
    ]

    for task_type, fname, title in [
        ("task",   "preprocessing_walk.png",   "Walk Window — Preprocessing"),
        ("stance", "preprocessing_stance.png", "Stance Window — Preprocessing"),
    ]:
        raw_df  = entry[task_type]["raw"]
        proc_df = entry[task_type]["proc"]
        sf      = entry[task_type]["sf"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row")
        fig.suptitle(title, fontweight="bold")

        col_labels   = ["Before Preprocessing", "After Preprocessing"]
        data_seq     = [(raw_df, sf), (proc_df, TARGET_HZ)]
        legend_handles, legend_labels = [], []

        for col_idx, (df, rate) in enumerate(data_seq):
            t = np.arange(len(df)) / rate
            for row_idx, (cols, sensor_label, unit) in enumerate(SENSOR_GROUPS):
                ax = axes[row_idx, col_idx]
                for col in cols:
                    axis_name = col.split("_")[1]
                    line, = ax.plot(t, df[col].values, color=CHAN_COLORS[axis_name],
                                    linewidth=0.8, label=axis_name)
                    if col_idx == 0 and row_idx == 0:
                        legend_handles.append(line)
                        legend_labels.append(axis_name)

                if row_idx == 0:
                    ax.set_title(col_labels[col_idx])
                if col_idx == 0:
                    ax.set_ylabel(f"{sensor_label}\n({unit})", fontweight="bold")
                if row_idx == 1:
                    ax.set_xlabel("Time (s)")

        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.02), framealpha=0.85,
                   title="Axis", title_fontsize=15)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {output_dir / fname}")

# =========================================================================
# 4. MAIN PIPELINE
# =========================================================================

def main():
    # keep only subjects that have both tasks 2 and 0 or 1
    all_processed_data = []
    clinical_map = {}

    for ds_name in DATASETS:
        cp = RAW_DATA_DIR / f"{ds_name}_clinical.csv"
        if cp.exists():
            cdf = pd.read_csv(cp)
            for _, r in cdf.iterrows():
                clinical_map[(ds_name, str(r["subjectID"]).strip())] = {
                    "postural_stability": r["postural_stability"],
                    "age": r.get("age", np.nan),
                    "gender": r.get("gender", np.nan),
                    "disease_duration": r.get("disease_duration", np.nan),
                }
    
    valid_subjects = set()
    for ds_name in DATASETS:
        sensor_path = RAW_DATA_DIR / f"{ds_name}_sensor.csv"
        if not sensor_path.exists(): continue
        
        df = pd.read_csv(sensor_path)
        df["subjectID"] = df["subjectID"].astype(str).str.strip()
        
        # Raggruppa per soggetto e trova i task unici svolti
        subject_tasks = df.groupby("subjectID")["taskID"].unique()
        
        for sid, tasks in subject_tasks.items():
            # Condizione: deve avere il 2 (Walk) E (0 oppure 1) (Stance)
            has_walk = 2 in tasks
            has_stance = 0 in tasks or 1 in tasks
            
            if has_walk and has_stance:
                # Verifica anche che il soggetto abbia un'etichetta clinica
                if (ds_name, sid) in clinical_map:
                    valid_subjects.add((ds_name, sid))

    print(f" {len(valid_subjects)} valid subjects found.")

    # (ds_name, sid) -> {"task": {raw, proc, sf}, "stance": {raw, proc, sf}}
    viz_collection = {}

    for ds_name in DATASETS:
        sensor_path = RAW_DATA_DIR / f"{ds_name}_sensor.csv"
        if not sensor_path.exists(): continue

        df = pd.read_csv(sensor_path)

        df["dataset"] = ds_name
        df["subjectID"] = df["subjectID"].astype(str).str.strip()

        # keep only subjects that have both tasks 2 and 0 or 1
        df = df[df["subjectID"].isin([sid for ds, sid in valid_subjects if ds == ds_name])]
        for (sid, tid), group in df.groupby(["subjectID", "taskID"]):
            if tid not in [0, 1, 2]: continue
            clinical_info = clinical_map.get((ds_name, sid))
            if clinical_info is None: continue
            label = clinical_info["postural_stability"]
            if pd.isna(label): continue

            sf = SF_DICT[ds_name]

            # Capture raw snapshot for each subject (first occurrence per task type)
            key_viz  = (ds_name, sid)
            task_type = "task" if tid == 2 else "stance"
            capture_viz = key_viz not in viz_collection or task_type not in viz_collection.get(key_viz, {})
            if capture_viz:
                n_raw = min(int(sf * WINDOW_SEC), len(group))
                viz_collection.setdefault(key_viz, {})[task_type] = {
                    "raw": group[SENSOR_COLS].iloc[:n_raw].reset_index(drop=True).copy(),
                    "sf": sf,
                }

            group = soft_trim_outliers(group, sf)
            # if taskID == 2 (Walk), apply bandpass; if taskID == 0 or 1 (Stance), apply lowpass
            if tid == 2:
                processed = apply_bandpass(group, sf)
            else:
                processed = apply_lowpass(group, sf)
            resampled = resample_group(processed, sf)

            n = len(resampled)
            trim = int(n * 0.05)
            if trim > 0:
                resampled = resampled.iloc[trim:n - trim].reset_index(drop=True)

            # Capture processed snapshot for the same window (after NaN interpolation)
            if capture_viz:
                n_proc = int(TARGET_HZ * WINDOW_SEC)
                proc_snippet = resampled[SENSOR_COLS].iloc[:n_proc].reset_index(drop=True).copy()
                for ch in SENSOR_COLS:
                    if proc_snippet[ch].isna().any():
                        proc_snippet[ch] = proc_snippet[ch].interpolate(method="linear").ffill().bfill()
                viz_collection[key_viz][task_type]["proc"] = proc_snippet

            resampled["label"] = merge_stability_label(label)
            resampled["age"] = clinical_info["age"]
            resampled["gender"] = clinical_info["gender"]
            resampled["disease_duration"] = clinical_info["disease_duration"]
            all_processed_data.append(resampled)

    if viz_collection:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_preprocessing_comparison(viz_collection, OUTPUT_DIR)
        if (VIZ_SUBJECT is not None
                and VIZ_SUBJECT in viz_collection
                and "task"   in viz_collection[VIZ_SUBJECT]
                and "stance" in viz_collection[VIZ_SUBJECT]):
            save_subject_pngs(viz_collection[VIZ_SUBJECT], OUTPUT_DIR)

    full_df = pd.concat(all_processed_data, ignore_index=True)    

    final_subject_tasks = full_df.groupby(["dataset", "subjectID"])["taskID"].unique()
    
    valid_final_subjects = []
    for (ds, sid), tasks in final_subject_tasks.items():
        has_walk = 2 in tasks
        has_stance = 0 in tasks or 1 in tasks
        if has_walk and has_stance:
            valid_final_subjects.append((ds, sid))
    
    full_df["ds_sid"] = list(zip(full_df["dataset"], full_df["subjectID"]))
    full_df = full_df[full_df["ds_sid"].isin(valid_final_subjects)]
    full_df = full_df.drop(columns=["ds_sid"])
    
    print(f"\n Subjects maintained after final filtering: {len(valid_final_subjects)}")
  
    # Generate aligned files with Adaptive Windowing
    x, y, meta = create_windows(full_df)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "windows.npy", x)
    np.save(OUTPUT_DIR / "labels.npy", y)
    meta.to_csv(OUTPUT_DIR / "metadata.csv", index=False)
    
    print("\n" + "="*50)
    print(f"PREPROCESSING COMPLETED (Adaptive Windowing [0.5 - 0.8])")
    print(f"Windows generated: {len(x)}")
    print(f"Class distribution:\n{meta['label'].value_counts().sort_index()}")
    print("="*50)

if __name__ == "__main__":
    main()