import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

# --- CONFIGURAZIONE ---
SF_DICT = {"fog_star": 60.0, "omnia_park": 90.0, "pd_phone": 200.0, "wearpd": 100.0, "kiel": 200.0}
DATASETS = ["fog_star", "omnia_park", "pd_phone", "wearpd", "kiel"]
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TARGET_HZ = 128
WINDOW_SEC = 5

# Parametri Adaptive Windowing
MIN_OVERLAP = 0.7
MAX_OVERLAP = 0.85
TARGET_CLASS_RATIO = 0.85  

RAW_DATA_DIR = Path("posturalInstability/data/cleaned_data")
OUTPUT_DIR = Path("posturalInstability/cnn_gru/data/")

# =========================================================================
# 1. FUNZIONI DI PULIZIA E FILTRAGGIO
# =========================================================================

def apply_lowpass(group, sf, cutoff=20.0):
    #print(f"  Applicazione filtro low-pass a {cutoff} Hz (sf={sf} Hz)")
    nyq = 0.5 * sf
    b, a = butter(4, cutoff / nyq, btype="low")
    for col in SENSOR_COLS:
        valid_mask = ~group[col].isna()
        if valid_mask.sum() > 30: 
            mean_val = group.loc[valid_mask, col].mean()
            centered = group.loc[valid_mask, col].values - mean_val
            group.loc[valid_mask, col] = filtfilt(b, a, centered) + mean_val
    return group

def soft_trim_outliers(group, sf, z_threshold=5):
    #   print(f"  Rimozione outlier con Z-score > {z_threshold} (sf={sf} Hz)")
    for col in SENSOR_COLS:
        rolling = group[col].rolling(window=int(sf*2), center=True, min_periods=1)
        z_score = np.abs((group[col] - rolling.mean()) / (rolling.std() + 1e-6))
        group.loc[z_score > z_threshold, col] = np.nan
    return group

# =========================================================================
# 2. LOGICA ADAPTIVE WINDOWING
# =========================================================================

def merge_stability_label(label):
    label = int(round(float(label)))
    return 3 if label >= 3 else label

def window_step(win_size, overlap):
    return max(1, int(round(win_size * (1 - overlap))))

def count_valid_windows(data, win_size, overlap):
    """Conta quante finestre valide (senza NaN) produrrebbe un segmento."""
    step = window_step(win_size, overlap)
    if len(data) < win_size: return 0
    count = 0
    for start_idx in range(0, len(data) - win_size + 1, step):
        if not np.isnan(data[start_idx:start_idx + win_size]).any():
            count += 1
    return count

def class_overlap_from_target(base_count, target_count):
    """Calcola l'overlap necessario per raggiungere il target di finestre."""
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
    x_orig = np.arange(len(group))
    x_new = np.linspace(0, len(group)-1, n_target)
    for col in SENSOR_COLS:
        resampled[col] = np.interp(x_new, x_orig, group[col].values)
    
    if "isTurn" in group.columns:
        turn_vals = group["isTurn"].values
        target_idx = np.linspace(0, len(group)-1, n_target).astype(int)
        resampled["isTurn"] = turn_vals[target_idx]
    else:
        resampled["isTurn"] = np.zeros(n_target)
    return pd.DataFrame(resampled)

def create_windows(df):
    windows, labels, metadata_rows = [], [], []
    win_size = int(TARGET_HZ * WINDOW_SEC)
    group_cols = ["subjectID", "sessionID", "dataset", "taskID", "label"]

    # --- FASE 1: Calcolo Base Counts ---
    print("Analisi distribuzione per Adaptive Windowing...")
    base_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for (_, _, _, _, label), group in df.groupby(group_cols):
        base_counts[label] += count_valid_windows(group[SENSOR_COLS].values, win_size, MIN_OVERLAP)

    majority_count = max(base_counts.values())
    target_count = int(majority_count * TARGET_CLASS_RATIO)
    
    # Calcolo overlap specifico per classe
    overlap_by_class = {
        cls: class_overlap_from_target(base_counts[cls], target_count) 
        for cls in base_counts
    }

    print(f"  Pianificazione overlap: { {k: round(v, 2) for k, v in overlap_by_class.items()} }")

    # --- FASE 2: Generazione Atomica ---
    window_id = 0
    for (sid, sessid, ds, tid, label), group in df.groupby(group_cols):
        data = group[SENSOR_COLS].values
        turns = group["isTurn"].values
        
        # Usa l'overlap calcolato per questa classe
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
# 3. PIPELINE PRINCIPALE
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
    
    # Generazione file allineati con Adaptive Windowing
    x, y, meta = create_windows(full_df)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "windows.npy", x)
    np.save(OUTPUT_DIR / "labels.npy", y)
    meta.to_csv(OUTPUT_DIR / "metadata.csv", index=False)
    
    print("\n" + "="*50)
    print(f"PREPROCESSING COMPLETATO (Adaptive Windowing [0.5 - 0.8])")
    print(f"Finestre generate: {len(x)}")
    print(f"Distribuzione classi:\n{meta['label'].value_counts().sort_index()}")
    print("="*50)

if __name__ == "__main__":
    main()