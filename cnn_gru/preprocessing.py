import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import os
from scipy.signal import butter, filtfilt
from sklearn.neighbors import NearestNeighbors

sf_dict = {"fog_star": 60.0, "omnia_park": 90.0, "pd_phone": 200.0, "wearpd": 100.0, "kiel": 200.0}
datasets = ["fog_star", "omnia_park", "pd_phone", "wearpd", "kiel"]
target_hz = 64
all_dfs = []

def enforce_nan_policy(group, sf, max_interp_gap_sec=0.2, sensor_cols=None):
    if sensor_cols is None:
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    group = group.sort_values('timestamp').copy()
    max_interp_gap = max(1, int(max_interp_gap_sec * sf))

    for col in sensor_cols:
        isna = group[col].isna().to_numpy()

        if isna.any():
            # Find contiguous NaN blocks and reject the whole session if one is too long.
            edges = np.diff(np.r_[False, isna, False].astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            max_gap = int((ends - starts).max())

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

    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
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

def lowpass_filter(group, sf, cutoff=15.0, order=4):
    nyquist = 0.5 * sf
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    for col in sensor_cols:
        mean_val = group[col].mean()
        signal_centered = group[col].values - mean_val
        filtered_centered = filtfilt(b, a, signal_centered)
        group[col] = filtered_centered + mean_val
        
    return group

def resample(group, original_sf, target_sf=64.0):
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    up, down = int(target_sf), int(original_sf)
    
    pad_samples = int(original_sf) 
    n_orig = len(group)
    n_target = int(n_orig * target_sf / original_sf)

    resampled_data = {
        'timestamp': np.linspace(0, (n_orig-1)/original_sf, n_target),
        'subjectID': group['subjectID'].iloc[0],
        'sessionID': group['sessionID'].iloc[0],
        'taskID': group['taskID'].iloc[0]
    }

    for col in sensor_cols:
        padded = np.pad(group[col].values, pad_width=pad_samples, mode='reflect')
        
        resampled_padded = resample_poly(padded, up, down)
        
        pad_target = int(pad_samples * target_sf / original_sf)
        resampled_data[col] = resampled_padded[pad_target : pad_target + n_target]
    
    return pd.DataFrame(resampled_data)


def create_windows(df, window_size, overlap):
    X = []
    meta = []
    step = int(window_size * (1 - overlap))
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Raggruppiamo per sessione reale per non mischiare i dati
    grouped = df.groupby(['subjectID', 'sessionID', 'taskID', 'dataset'])
    
    for (sub_id, sess_id, task_id, dataset), group in grouped:
        data = group[sensor_cols].values
        if len(data) < window_size:
            continue
            
        for i in range(0, len(data) - window_size + 1, step):
            X.append(data[i : i + window_size])
            meta.append({
                'subjectID': sub_id,
                'sessionID': sess_id,
                'taskID': task_id,
                'dataset': dataset
            })
            
    return np.array(X), pd.DataFrame(meta)

def oversample(windows, labels, target_count, k_neighbors=3):

    unique_labels = np.unique(labels)
    new_windows = []
    new_labels = []
    
    rng = np.random.default_rng(42)

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        cls_windows = windows[idx]
        
        # Se la classe ha già abbastanza campioni, la usiamo così com'è 
        # o facciamo un leggero downsampling se vogliamo pareggiare a target_count
        if len(cls_windows) >= target_count:
            chosen_idx = rng.choice(idx, size=target_count, replace=False)
            new_windows.append(windows[chosen_idx])
            new_labels.append(labels[chosen_idx])
            continue

        new_windows.append(cls_windows) # Teniamo gli originali
        new_labels.append(labels[idx])
        
        flat_windows = cls_windows.reshape(len(cls_windows), -1)
        nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(cls_windows)), metric="euclidean")
        nn.fit(flat_windows)
        knns = nn.kneighbors(flat_windows, return_distance=False)

        synth_windows = []
        num_to_add = target_count - len(cls_windows)
        
        for _ in range(num_to_add):

            i = rng.integers(0, len(cls_windows))
            neighbor_idx = rng.choice(knns[i][1:]) # Escludiamo se stesso
            
            # Interpolazione lineare (SMOTE): crea una finestra "in mezzo" alle due
            alpha = rng.random()
            synthetic_sample = cls_windows[i] + alpha * (cls_windows[neighbor_idx] - cls_windows[i])
            
            # Aggiungiamo un leggero Jittering (rumore) come suggerito per la robustezza
            noise = rng.normal(0, 0.001, synthetic_sample.shape)
            synth_windows.append(synthetic_sample + noise)
            
        new_windows.append(np.stack(synth_windows))
        new_labels.append(np.full(num_to_add, label))

    return np.concatenate(new_windows), np.concatenate(new_labels)


def oversample_dataset_subset(windows, labels, metadata, k_neighbors=3):
    """Balance class counts within a single dataset using the majority class count.

    Returns oversampled windows, labels, and metadata for that dataset.
    """
    labels = np.asarray(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 0:
        return windows, labels, metadata

    target_count = int(counts.max())
    balanced_windows = []
    balanced_labels = []
    balanced_metadata = []

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        cls_windows = windows[idx]
        cls_metadata = metadata.iloc[idx].reset_index(drop=True)

        if len(cls_windows) >= target_count:
            chosen_idx = np.random.default_rng(42).choice(len(cls_windows), size=target_count, replace=False)
            balanced_windows.append(cls_windows[chosen_idx])
            balanced_labels.append(np.full(target_count, label, dtype=np.int64))
            balanced_metadata.append(cls_metadata.iloc[chosen_idx].reset_index(drop=True))
            continue

        flat_windows = cls_windows.reshape(len(cls_windows), -1)
        nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(cls_windows)), metric="euclidean")
        nn.fit(flat_windows)
        knns = nn.kneighbors(flat_windows, return_distance=False)

        synth_windows = []
        synth_metadata = []
        num_to_add = target_count - len(cls_windows)
        rng = np.random.default_rng(42)

        for _ in range(num_to_add):
            i = rng.integers(0, len(cls_windows))
            neighbor_candidates = knns[i][1:]
            if len(neighbor_candidates) == 0:
                neighbor_idx = i
            else:
                neighbor_idx = rng.choice(neighbor_candidates)

            alpha = rng.random()
            synthetic_sample = cls_windows[i] + alpha * (cls_windows[neighbor_idx] - cls_windows[i])
            noise = rng.normal(0, 0.001, synthetic_sample.shape)
            synth_windows.append(synthetic_sample + noise)
            synth_metadata.append(cls_metadata.iloc[i].to_dict())

        balanced_windows.append(np.concatenate([cls_windows, np.stack(synth_windows)]))
        balanced_labels.append(np.full(target_count, label, dtype=np.int64))
        balanced_metadata.append(pd.concat([cls_metadata, pd.DataFrame(synth_metadata)], ignore_index=True))

    return (
        np.concatenate(balanced_windows),
        np.concatenate(balanced_labels),
        pd.concat(balanced_metadata, ignore_index=True),
    )

sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

for dataset in datasets:
    path = f"posturalInstability/data/cleaned_data/{dataset}_sensor.csv"
    if not os.path.exists(path):
        
        print(f"File not found: {path}. Skipping {dataset}.")
        continue

    print(f"\n--- Processing {dataset} ---")
    df = pd.read_csv(path)
    sf = sf_dict[dataset]

    df_filtered = df[df.taskID.isin([0, 1, 2])].copy()
    processed_sessions = []
    total_count, discarded_count = 0, 0

    grouped = df_filtered.groupby(['subjectID', 'sessionID', 'taskID'])

    for _, session_group in grouped:
        total_count += 1

        if session_group['taskID'].iloc[0] in [0, 1]:
            # Task 0,1: trim outliers + strict NaN policy + lowpass + resample
            temp_group = trim_outliers(session_group, sf)
            if temp_group is None:
                discarded_count += 1
                continue
        else:
            # Task 2: no outlier trim, but same strict NaN policy
            temp_group = session_group.sort_values('timestamp').iloc[int(sf):-int(sf)].copy()
            if len(temp_group) <= sf:
                discarded_count += 1
                continue

            temp_group = enforce_nan_policy(temp_group, sf, max_interp_gap_sec=0.2, sensor_cols=sensor_cols)
            if temp_group is None:
                discarded_count += 1
                continue

            temp_group['timestamp'] = (temp_group['timestamp'] - temp_group['timestamp'].min()).round(4)

        temp_group = lowpass_filter(temp_group, sf, cutoff=15.0)
        temp_group = resample(temp_group, sf, target_sf=target_hz)

        # Final hard check: never keep sessions with NaNs in sensor features.
        if temp_group[sensor_cols].isna().any().any():
            discarded_count += 1
            continue

        processed_sessions.append(temp_group)

    if processed_sessions:
        out_df = pd.concat(processed_sessions, ignore_index=True)
        out_df["dataset"] = dataset

        # Keep only fully usable processed sessions (session-level NaN-free check).
        valid_mask = (
            out_df.groupby(['subjectID', 'sessionID', 'taskID'])[sensor_cols]
            .transform(lambda x: ~x.isna().any())
            .all(axis=1)
        )
        out_df = out_df.loc[valid_mask].copy()
        out_df = out_df.dropna(subset=sensor_cols)

        all_dfs.append(out_df)
        print(f"Saved: {total_count - discarded_count}/{total_count} sessions.")
        print(f"Final usable rows: {len(out_df)}")
    else:
        print(f"No valid sessions for {dataset} after quality checks.")

if not all_dfs:
    raise ValueError(
        "No preprocessed sessions were generated. Check the cleaned_data paths and quality filters."
    )

full_df = pd.concat(all_dfs, ignore_index=True)

print("Create windows...")
X_raw, metadata = create_windows(full_df, window_size=target_hz*5, overlap=0.75)
n_windows, w_size, n_channels = X_raw.shape
X_flat = X_raw.reshape(-1, n_channels)
X_final = X_flat.reshape(n_windows, w_size, n_channels)

# Load clinical labels per dataset to map subject -> postural_stability
clinical_map = {}
for ds in datasets:
    clin_path = f"posturalInstability/data/cleaned_data/{ds}_clinical.csv"
    if not os.path.exists(clin_path):
        continue
    clin_df = pd.read_csv(clin_path, dtype={"subjectID": str}, low_memory=False)
    clin_df["subjectID"] = clin_df["subjectID"].astype(str).str.strip()
    if "postural_stability" in clin_df.columns:
        for _, row in clin_df.iterrows():
            key = (ds, row["subjectID"])
            try:
                val = float(row["postural_stability"]) if not pd.isna(row["postural_stability"]) else np.nan
            except Exception:
                val = np.nan
            clinical_map[key] = val

# Build labels array aligned with metadata
labels = []
for _, row in metadata.iterrows():
    ds = row["dataset"]
    sid = str(row["subjectID"]).strip()
    lbl = clinical_map.get((ds, sid), np.nan)
    labels.append(lbl)

labels = np.array(labels, dtype=np.float32)

# Filter out windows without clinical label (NaN)
has_label_mask = ~np.isnan(labels)
num_total = len(labels)
num_labeled = int(has_label_mask.sum())
print(f"Total windows: {num_total}, labeled windows: {num_labeled}")

os.makedirs("posturalInstability/cnn_gru/data/windowed_data", exist_ok=True)
if num_labeled == 0:
    print("Warning: no labeled windows found.")
else:
    X_final = X_final[has_label_mask]
    labels = labels[has_label_mask]
    metadata = metadata.loc[has_label_mask].reset_index(drop=True)

    # Convert labels to integers (postural_stability typically 0..4)
    labels = np.nan_to_num(labels, nan=-1.0)
    labels = np.clip(labels, 0, 4).astype(np.int64)

    # Oversample independently inside each dataset.
    oversampled_windows = []
    oversampled_labels = []
    oversampled_metadata = []
    for dataset in datasets:
        dataset_mask = metadata["dataset"] == dataset
        if not dataset_mask.any():
            continue

        ds_windows = X_final[dataset_mask.to_numpy()]
        ds_labels = labels[dataset_mask.to_numpy()]
        ds_metadata = metadata.loc[dataset_mask].reset_index(drop=True)

        balanced_windows, balanced_labels, balanced_metadata = oversample_dataset_subset(
            ds_windows,
            ds_labels,
            ds_metadata,
            k_neighbors=3,
        )

        oversampled_windows.append(balanced_windows)
        oversampled_labels.append(balanced_labels)
        oversampled_metadata.append(balanced_metadata)

    if oversampled_windows:
        X_final = np.concatenate(oversampled_windows, axis=0)
        labels = np.concatenate(oversampled_labels, axis=0)
        metadata = pd.concat(oversampled_metadata, ignore_index=True)

    # Shuffle after per-dataset balancing so datasets are mixed in the final file.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(labels))
    X_final = X_final[perm]
    labels = labels[perm]
    metadata = metadata.iloc[perm].reset_index(drop=True)

    np.save("posturalInstability/cnn_gru/data/windowed_data/windows.npy", X_final.astype(np.float32))
    np.save("posturalInstability/cnn_gru/data/windowed_data/labels.npy", labels)
    metadata.to_csv("posturalInstability/cnn_gru/data/windowed_data/metadata.csv", index=False)
    print(f"Saved {len(labels)} labeled windows.")