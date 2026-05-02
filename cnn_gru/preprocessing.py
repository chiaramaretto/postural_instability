import numpy as np
import pandas as pd
import os
from scipy.signal import resample_poly, butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Speed-up rendering for large vector paths and simplify paths where possible
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 0.5


SF_DICT = {"fog_star": 60.0, "omnia_park": 90.0, "pd_phone": 200.0, "wearpd": 100.0, "kiel": 200.0}
DATASETS = ["fog_star", "omnia_park", "pd_phone", "wearpd", "kiel"]
SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
TARGET_HZ = 64
WINDOW_SEC = 10
OVERLAP = 0

import matplotlib.pyplot as plt

def plot_comparison(original, processed, ds_name, subject_id, task_id, pdf):
    # Creiamo 2 righe (Acc, Gyro) e 2 colonne (Original, Processed)
    print(f"Plotting comparison for Dataset: {ds_name}, Subject: {subject_id}, Task: {task_id}")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex='col')
    fig.suptitle(f"Dataset: {ds_name} | Subject: {subject_id} | Task: {task_id}", fontsize=16, fontweight='bold')
    
    titles = ["Original Data (Cleaned)", f"Processed Data ({TARGET_HZ}Hz + Lowpass)"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']

    # Configuriamo i titoli delle colonne
    for j in range(2):
        axes[0, j].set_title(titles[j], fontsize=14, pad=15)

    # --- ROW 0: ACCELEROMETER ---
    for i, col in enumerate(acc_cols):
        # Original (Sinistra)
        axes[0, 0].plot(original['timestamp'], original[col], label=col, 
                        color=colors[i], alpha=0.8, linewidth=1)
        # Processed (Destra)
        axes[0, 1].plot(processed['timestamp'], processed[col], label=col, 
                        color=colors[i], alpha=0.8, linewidth=1, rasterized=True)
    
    axes[0, 0].set_ylabel("Acceleration [g o m/s²]", fontsize=12)
    axes[0, 0].legend(loc='upper right', fontsize=10)
    axes[0, 1].legend(loc='upper right', fontsize=10)

    # --- ROW 1: GYROSCOPE ---
    for i, col in enumerate(gyro_cols):
        # Original (Sinistra)
        axes[1, 0].plot(original['timestamp'], original[col], label=col, 
                        color=colors[i], alpha=0.8, linewidth=1)
        # Processed (Destra)
        axes[1, 1].plot(processed['timestamp'], processed[col], label=col, 
                        color=colors[i], alpha=0.8, linewidth=1, rasterized=True)

    axes[1, 0].set_ylabel("Angular Velocity [rad/s]", fontsize=12)
    axes[1, 0].set_xlabel("Time [s]", fontsize=12)
    axes[1, 1].set_xlabel("Time [s]", fontsize=12)
    axes[1, 0].legend(loc='upper right', fontsize=10)
    axes[1, 1].legend(loc='upper right', fontsize=10)

    # Griglia e layout
    for ax in axes.flat:
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig, dpi=80)
    plt.close(fig)

def enforce_nan_policy(group, sf, max_interp_gap_sec=0.2, sensor_cols=None):
    if sensor_cols is None:
        sensor_cols = SENSOR_COLS

    group = group.sort_values('timestamp').copy()
    max_interp_gap = max(1, int(max_interp_gap_sec * sf))

    for col in sensor_cols:
        isna = group[col].isna().to_numpy()

        if isna.any():
            # Find contiguous NaN blocks and reject the whole session if one is too long.
            edges = np.diff(np.r_[False, isna, False].astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            gap_lengths = ends - starts
            max_gap = int(gap_lengths.max()) if len(gap_lengths) else 0

            if max_gap > max_interp_gap:
                return None

            group[col] = group[col].interpolate(method='linear', limit_direction='both')

        # Safety check: if interpolation could not fill everything, reject session.
        if group[col].isna().any():
            return None

    return group


def trim_outliers(group, sf, z_threshold=2.5, trim_perc=0.15, window_size_sec=3.0):
    group = group.sort_values('timestamp').copy()

    total_duration = group['timestamp'].max() - group['timestamp'].min()
    n_trim = int(trim_perc * total_duration * sf)

    if len(group) < (2 * n_trim + sf):
        return None
    group = group.iloc[n_trim:-n_trim].copy()

    sensor_cols = SENSOR_COLS
    window_size = max(3, int(window_size_sec * sf))

    for col in sensor_cols:
        rolling = group[col].rolling(window=window_size, center=True, min_periods=1)
        z_score = np.abs((group[col] - rolling.mean()) / (rolling.std() + 1e-6))
        group.loc[z_score > z_threshold, col] = np.nan

    # Apply strict NaN policy after outlier removal.
    group = enforce_nan_policy(group, sf, max_interp_gap_sec=0.2, sensor_cols=sensor_cols)
    if group is None:
        return None

    group['timestamp'] = (group['timestamp'] - group['timestamp'].min()).round(4)
    return group

def apply_lowpass(group, sf, cutoff=20.0):
    nyq = 0.5 * sf
    b, a = butter(4, cutoff / nyq, btype='low')
    for col in SENSOR_COLS:
        group[col] = filtfilt(b, a, group[col].values - group[col].mean()) + group[col].mean()
    return group

def resample_group(group, original_sf):
    up, down = int(TARGET_HZ), int(original_sf)
    n_target = int(len(group) * TARGET_HZ / original_sf)
    
    res = {
        'timestamp': np.linspace(0, (len(group)-1)/original_sf, n_target),
        'subjectID': group['subjectID'].iloc[0],
        'sessionID': group['sessionID'].iloc[0],
        'taskID': group['taskID'].iloc[0]
    }
    
    for col in SENSOR_COLS:
        padded = np.pad(group[col].values, (int(original_sf), int(original_sf)), mode='reflect')
        resampled = resample_poly(padded, up, down)
        res[col] = resampled[up : up + n_target]
    
    return pd.DataFrame(res)

def create_windows(df):
    X, meta = [], []
    win_size = int(TARGET_HZ * WINDOW_SEC)
    step = int(win_size * (1 - OVERLAP))
    
    for (sid, ds, tid), group in df.groupby(['subjectID', 'dataset', 'taskID']):
        data = group[SENSOR_COLS].values
        if len(data) < win_size: continue
        
        for i in range(0, len(data) - win_size + 1, step):
            X.append(data[i : i + win_size])
            meta.append({
                'subjectID': sid, 
                'dataset': ds, 
                'taskID': tid  
            })
            
    return np.array(X), pd.DataFrame(meta)

all_dfs = []
pdf_path = "posturalInstability/preprocessing_report.pdf"

with PdfPages(pdf_path) as pdf:
    for ds in DATASETS:
        path = f"posturalInstability/data/cleaned_data/{ds}_sensor.csv"
        if not os.path.exists(path): continue
        
        df = pd.read_csv(path)
        sf = SF_DICT[ds]
        df = df[df.taskID.isin([0, 1, 2])].copy()
        
        for (sid, sessid, tid), group in df.groupby(['subjectID', 'sessionID', 'taskID']):
            if tid in [0, 1]:
                # Tasks 0/1: trim outliers first, then enforce the NaN policy.
                base_group = trim_outliers(group, sf)
                if base_group is None:
                    continue
            else:
                # Task 2: no outlier trimming, but still discard records with invalid NaN gaps.
                base_group = group.sort_values('timestamp').iloc[int(sf):-int(sf)].copy()
                if len(base_group) <= sf:
                    continue
                base_group = enforce_nan_policy(base_group, sf, max_interp_gap_sec=0.2, sensor_cols=SENSOR_COLS)
                if base_group is None:
                    continue

            base_group = enforce_nan_policy(base_group, sf, max_interp_gap_sec=0.2, sensor_cols=SENSOR_COLS)
            if base_group is None:
                continue

            original_signal = base_group.copy()
            original_signal['timestamp'] = (original_signal['timestamp'] - original_signal['timestamp'].iloc[0])
            processed = apply_lowpass(base_group, sf)
            processed = resample_group(processed, sf)
            processed['dataset'] = ds
            
            plot_comparison(original_signal, processed, ds, sid, tid, pdf)
            
            all_dfs.append(processed)

full_df = pd.concat(all_dfs, ignore_index=True)
X_raw, meta_df = create_windows(full_df)

clinical_map = {}
for ds in DATASETS:
    c_path = f"posturalInstability/data/cleaned_data/{ds}_clinical.csv"
    if not os.path.exists(c_path): continue
    cdf = pd.read_csv(c_path, dtype={"subjectID": str})
    if "postural_stability" in cdf.columns:
        for _, r in cdf.iterrows():
            clinical_map[(ds, str(r["subjectID"]).strip())] = r["postural_stability"]

labels = []
valid_indices = []
for i, r in meta_df.iterrows():
    lbl = clinical_map.get((r["dataset"], str(r["subjectID"]).strip()), np.nan)
    if not pd.isna(lbl):
        labels.append(int(np.clip(lbl, 0, 4)))
        valid_indices.append(i)

X_final = X_raw[valid_indices].astype(np.float32)
labels_final = np.array(labels, dtype=np.int64)
meta_final = meta_df.iloc[valid_indices].reset_index(drop=True)

labels_final = np.where(labels_final == 4, 3, labels_final)

save_path = "posturalInstability/cnn_gru/data/windowed_data"
os.makedirs(save_path, exist_ok=True)
np.save(f"{save_path}/windows.npy", X_final)
np.save(f"{save_path}/labels.npy", labels_final)
meta_final.to_csv(f"{save_path}/metadata.csv", index=False)