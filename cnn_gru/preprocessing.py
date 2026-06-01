import os
import numpy as np
import pandas as pd
from pathlib import Path
from fractions import Fraction
from scipy.signal import butter, filtfilt, resample_poly

# --- CONFIGURATION ---
SF_DICT = {"fog_star": 60.0, "omnia_park": 90.0, "pd_phone": 200.0, "wearpd": 100.0, "kiel": 200.0}
DATASETS = ["fog_star", "omnia_park", "pd_phone", "wearpd", "kiel"]
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TARGET_HZ = 64
WINDOW_SEC = 5

# Adaptive Windowing Parameters
MIN_OVERLAP = 0.7
MAX_OVERLAP = 0.85
TARGET_CLASS_RATIO = 0.85  

RAW_DATA_DIR = Path("posturalInstability/data/cleaned_data")
OUTPUT_DIR = Path("posturalInstability/cnn_gru/data/")

# =========================================================================
# 1. CLEANING AND FILTERING FUNCTIONS
# =========================================================================

def apply_lowpass(group, sf, cutoff=20.0):
    #print(f"  Applying low-pass filter at {cutoff} Hz (sf={sf} Hz)")
    nyq = 0.5 * sf
    b, a = butter(4, cutoff / nyq, btype="low")
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

def create_windows(df):
    windows, labels, metadata_rows = [], [], []
    win_size = int(TARGET_HZ * WINDOW_SEC)
    group_cols = ["subjectID", "sessionID", "dataset", "taskID", "label"]

    # --- PHASE 1: Compute Base Counts ---
    print("Analyzing distribution for Adaptive Windowing...")
    base_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for (_, _, _, _, label), group in df.groupby(group_cols):
        base_counts[label] += count_valid_windows(group[SENSOR_COLS].values, win_size, MIN_OVERLAP)

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
        
        # Use the overlap computed for this class
        current_overlap = overlap_by_class[label]
        step = window_step(win_size, current_overlap)
        
        for start_idx in range(0, len(data) - win_size + 1, step):
            win_data = data[start_idx : start_idx + win_size]
            if np.isnan(win_data).any(): continue

            windows.append(win_data)
            labels.append(label)
            metadata_rows.append({
                "window_id": window_id,
                "subjectID": sid,
                "dataset": ds,
                "taskID": tid,
                "isTurn": int(turns[start_idx : start_idx + win_size].max()),
                "label": label,
                "overlap": current_overlap
            })
            window_id += 1

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32), pd.DataFrame(metadata_rows)

# =========================================================================
# 3. MAIN PIPELINE
# =========================================================================

def main():
    all_processed_data = []
    clinical_map = {}

    for ds_name in DATASETS:
        cp = RAW_DATA_DIR / f"{ds_name}_clinical.csv"
        if cp.exists():
            cdf = pd.read_csv(cp)
            for _, r in cdf.iterrows():
                clinical_map[(ds_name, str(r["subjectID"]).strip())] = r["postural_stability"]

    for ds_name in DATASETS:
        sensor_path = RAW_DATA_DIR / f"{ds_name}_sensor.csv"
        if not sensor_path.exists(): continue
        
        df = pd.read_csv(sensor_path)
        df["dataset"] = ds_name
        df["subjectID"] = df["subjectID"].astype(str).str.strip()

        for (sid, tid), group in df.groupby(["subjectID", "taskID"]):
            if tid not in [0, 1, 2]: continue
            label = clinical_map.get((ds_name, sid), np.nan)
            if np.isnan(label): continue
            
            sf = SF_DICT[ds_name]

            group = soft_trim_outliers(group, sf)            
            processed = apply_lowpass(group, sf)
            resampled = resample_group(processed, sf)
            resampled["label"] = merge_stability_label(label)
            all_processed_data.append(resampled)

    full_df = pd.concat(all_processed_data, ignore_index=True)
    
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