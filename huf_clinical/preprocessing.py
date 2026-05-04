import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.model_selection import train_test_split

# --- Configuration ---
SF_DICT = {"fog_star": 60.0, "omnia_park": 90.0, "pd_phone": 200.0, "wearpd": 100.0, "kiel": 200.0}
DATASETS = ["fog_star", "omnia_park", "pd_phone", "wearpd", "kiel"]
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TARGET_HZ = 128.0
WINDOW_SEC = 5
MIN_OVERLAP = 0.2  # Overlap minimo richiesto

RAW_DATA_DIR = Path("posturalInstability/data/cleaned_data")
BASE_OUTPUT_DIR = Path("posturalInstability/huf_clinical/data/windowed_data")
CHANNELS_OUTPUT_DIR = BASE_OUTPUT_DIR / "channels"

# --- Preprocessing Functions ---

def enforce_nan_policy(group, sf, max_interp_gap_sec=0.2):
    group = group.sort_values("timestamp").copy()
    max_interp_gap = max(1, int(max_interp_gap_sec * sf))
    for col in SENSOR_COLS:
        isna = group[col].isna().to_numpy()
        if isna.any():
            edges = np.diff(np.r_[False, isna, False].astype(int))
            starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
            if len(starts) > 0 and (ends - starts).max() > max_interp_gap:
                return None
            group[col] = group[col].interpolate(method="linear", limit_direction="both")
    return None if group[SENSOR_COLS].isna().any().any() else group

def trim_outliers(group, sf, z_threshold=2.5, trim_perc=0.15):
    group = group.sort_values("timestamp").copy()
    n_trim = int(trim_perc * (group["timestamp"].max() - group["timestamp"].min()) * sf)
    if len(group) < (2 * n_trim + sf): return None
    group = group.iloc[n_trim:-n_trim].copy()
    window_size = max(3, int(3.0 * sf))
    for col in SENSOR_COLS:
        rolling = group[col].rolling(window=window_size, center=True, min_periods=1)
        z_score = np.abs((group[col] - rolling.mean()) / (rolling.std() + 1e-6))
        group.loc[z_score > z_threshold, col] = np.nan
    return enforce_nan_policy(group, sf)

def apply_lowpass(group, sf, cutoff=15.0):
    nyq = 0.5 * sf
    b, a = butter(4, cutoff / nyq, btype="low")
    for col in SENSOR_COLS:
        group[col] = filtfilt(b, a, group[col].values - group[col].mean()) + group[col].mean()
    return group

def resample_group(group, original_sf):
    n_target = int(len(group) * TARGET_HZ / original_sf)
    resampled = {
        "timestamp": np.linspace(0, (len(group)-1)/original_sf, n_target),
        "subjectID": group["subjectID"].iloc[0],
        "taskID": int(group["taskID"].iloc[0]),
        "dataset": group["dataset"].iloc[0] if "dataset" in group.columns else "unknown"
    }
    up, down = int(TARGET_HZ), int(original_sf)
    for col in SENSOR_COLS:
        padded = np.pad(group[col].values, int(original_sf), mode="reflect")
        sig = resample_poly(padded, up, down)
        resampled[col] = sig[up : up + n_target]
    return pd.DataFrame(resampled)

# --- Adattamento Overlap ---

def choose_label_overlaps(df, target_fraction=0.8):
    """Calcola l'overlap necessario per bilanciare le classi."""
    win_size = int(TARGET_HZ * WINDOW_SEC)
    labels = sorted(df["label"].unique())
    
    # Conta finestre con overlap minimo (0.2)
    counts = {}
    for label in labels:
        label_df = df[df["label"] == label]
        total_windows = 0
        step = int(win_size * (1 - MIN_OVERLAP))
        for _, group in label_df.groupby(["subjectID", "dataset", "taskID"]):
            if len(group) >= win_size:
                total_windows += 1 + (len(group) - win_size) // step
        counts[label] = total_windows

    majority_label = max(counts, key=counts.get)
    target_count = int(counts[majority_label] * target_fraction)
    
    overlap_by_label = {}
    for label in labels:
        if label == majority_label:
            overlap_by_label[label] = MIN_OVERLAP
            continue
        
        # Cerca l'overlap tra MIN_OVERLAP e 0.95 per raggiungere il target_count
        best_ov = MIN_OVERLAP
        for ov in np.linspace(MIN_OVERLAP, 0.95, 16):
            step = max(1, int(win_size * (1 - ov)))
            current_windows = 0
            label_df = df[df["label"] == label]
            for _, group in label_df.groupby(["subjectID", "dataset", "taskID"]):
                if len(group) >= win_size:
                    current_windows += 1 + (len(group) - win_size) // step
            if current_windows >= target_count:
                best_ov = ov
                break
        overlap_by_label[label] = float(best_ov)
        
    return overlap_by_label

# --- Windowing & Split ---

def create_windows(df, overlap_by_label):
    X, meta = [], []
    win_size = int(TARGET_HZ * WINDOW_SEC)
    
    for (sid, ds, tid, label), group in df.groupby(["subjectID", "dataset", "taskID", "label"]):
        ov = overlap_by_label.get(label, MIN_OVERLAP)
        step = max(1, int(win_size * (1 - ov)))
        data = group[SENSOR_COLS].values
        
        if len(data) < win_size: continue
        for i in range(0, len(data) - win_size + 1, step):
            X.append(data[i : i + win_size])
            meta.append({"subjectID": sid, "dataset": ds, "taskID": tid, "label": int(label)})
            
    return np.array(X), pd.DataFrame(meta)

def save_huf_split(name, X, y, meta):
    split_dir = BASE_OUTPUT_DIR / name
    chan_dir = CHANNELS_OUTPUT_DIR / name
    for d in [split_dir, chan_dir]: d.mkdir(parents=True, exist_ok=True)
    
    np.save(split_dir / "windows.npy", X.astype(np.float32))
    np.save(split_dir / "labels.npy", y.astype(np.int64))
    meta.to_csv(split_dir / "metadata.csv", index=False)
    
    for i, col in enumerate(SENSOR_COLS):
        np.save(chan_dir / f"{col}.npy", X[:, :, i].astype(np.float32))

def main():
    all_data = []
    clinical_map = {}
    for ds in DATASETS:
        cp = RAW_DATA_DIR / f"{ds}_clinical.csv"
        if cp.exists():
            cdf = pd.read_csv(cp, dtype={"subjectID": str})
            if "postural_stability" in cdf.columns:
                for _, r in cdf.iterrows():
                    clinical_map[(ds, str(r["subjectID"]).strip())] = r["postural_stability"]

    for ds in DATASETS:
        sp = RAW_DATA_DIR / f"{ds}_sensor.csv"
        if not sp.exists(): continue
        df = pd.read_csv(sp, dtype={"subjectID": str})
        df = df[df["taskID"].isin([0, 1, 2])].copy()
        df["dataset"] = ds
        
        for _, group in df.groupby(["subjectID", "sessionID", "taskID"]):
            sf = SF_DICT[ds]
            proc = trim_outliers(group, sf) if group["taskID"].iloc[0] in [0,1] else enforce_nan_policy(group, sf)
            if proc is not None:
                proc = apply_lowpass(proc, sf)
                all_data.append(resample_group(proc, sf))

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Mapping label cliniche
    full_df["label"] = full_df.apply(lambda r: clinical_map.get((r["dataset"], str(r["subjectID"]).strip()), np.nan), axis=1)
    full_df = full_df.dropna(subset=["label"]).copy()
    full_df["label"] = np.clip(full_df["label"].astype(int), 0, 3)

    # 1. Calcolo Overlap Adattabile
    overlap_by_label = choose_label_overlaps(full_df)
    print(f"Overlap calcolato per classe: {overlap_by_label}")

    # 2. Creazione Finestre
    X_raw, meta = create_windows(full_df, overlap_by_label)
    y = meta["label"].values
    
    # 3. Split Subject-wise (Corretto)
    subjs = meta.apply(lambda r: (r['dataset'], str(r['subjectID']).strip()), axis=1)
    unique_subjs = subjs.unique()
    
    train_s, test_s = train_test_split(unique_subjs, test_size=0.2, random_state=42)
    train_s, val_s = train_test_split(train_s, test_size=0.125, random_state=42)

    # Trasformo in set per evitare il ValueError di broadcasting con NumPy
    train_set, val_set, test_set = set(train_s), set(val_s), set(test_s)

    for name, s_set in [("train", train_set), ("val", val_set), ("test", test_set)]:
        # Controllo appartenenza usando set (veloce e sicuro)
        idx = [i for i, s in enumerate(subjs) if s in s_set]
        if idx:
            save_huf_split(name, X_raw[idx], y[idx], meta.iloc[idx])

    print(f"Preprocessing complete. Finestre totali: {len(X_raw)}")

if __name__ == "__main__":
    main()