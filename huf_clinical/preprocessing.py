import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, butter, filtfilt
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# Configuration
sf_dict = {"fog_star": 60.0, "omnia_park": 90.0, "pd_phone": 200.0, "wearpd": 100.0, "kiel": 200.0}
datasets = ["fog_star", "omnia_park", "pd_phone", "wearpd", "kiel"]
target_hz = 128.0  # Target sampling rate for huf_clinical
all_dfs = []


def enforce_nan_policy(group, sf, max_interp_gap_sec=0.2, sensor_cols=None):
    """Enforce strict NaN handling: reject sessions with gaps > max_interp_gap_sec."""
    if sensor_cols is None:
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    group = group.sort_values('timestamp').copy()
    max_interp_gap = max(1, int(max_interp_gap_sec * sf))

    for col in sensor_cols:
        isna = group[col].isna().to_numpy()
        if isna.any():
            edges = np.diff(np.r_[False, isna, False].astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            max_gap = int((ends - starts).max()) if len(starts) > 0 else 0
            if max_gap > max_interp_gap:
                return None
            group[col] = group[col].interpolate(method='linear', limit_direction='both')
        if group[col].isna().any():
            return None
    return group


def trim_outliers(group, sf, z_threshold=2.5, trim_perc=0.15, window_size_sec=3.0):
    """Trim edges and remove outliers using z-score."""
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

    group = enforce_nan_policy(group, sf, max_interp_gap_sec=0.2, sensor_cols=sensor_cols)
    if group is None:
        return None

    group['timestamp'] = (group['timestamp'] - group['timestamp'].min()).round(4)
    return group


def lowpass_filter(group, sf, cutoff=15.0, order=4):
    """Apply lowpass filter to sensor columns."""
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


def resample(group, original_sf, target_sf=128.0):
    """Resample sensor data to target sampling rate."""
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    up, down = int(target_sf), int(original_sf)
    pad_samples = int(original_sf)
    n_orig = len(group)
    n_target = int(n_orig * target_sf / original_sf)

    resampled_data = {
        'timestamp': np.linspace(0, (n_orig - 1) / original_sf, n_target),
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
    """Create sliding windows from preprocessed data."""
    X = []
    meta = []
    step = int(window_size * (1 - overlap))
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

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


def oversample_dataset_subset(windows_arr, labels_arr, metadata_df, k_neighbors=3):
    """Oversample minority classes within a dataset using SMOTE-like approach."""
    labels_arr = np.asarray(labels_arr)
    unique_labels, counts = np.unique(labels_arr, return_counts=True)
    if len(unique_labels) == 0:
        return windows_arr, labels_arr, metadata_df

    target_count = int(counts.max())
    balanced_windows = []
    balanced_labels = []
    balanced_metadata = []

    for label in unique_labels:
        idx = np.where(labels_arr == label)[0]
        cls_windows = windows_arr[idx]
        cls_metadata = metadata_df.iloc[idx].reset_index(drop=True)

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


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# Step 1: Load and preprocess data per dataset
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
            temp_group = trim_outliers(session_group, sf)
            if temp_group is None:
                discarded_count += 1
                continue
        else:
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

        if temp_group[sensor_cols].isna().any().any():
            discarded_count += 1
            continue

        processed_sessions.append(temp_group)

    if processed_sessions:
        out_df = pd.concat(processed_sessions, ignore_index=True)
        out_df["dataset"] = dataset

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
    raise ValueError("No preprocessed sessions were generated.")

full_df = pd.concat(all_dfs, ignore_index=True)

# Step 2: Create windows (ordered, no global shuffle)
print("\nCreating windows...")
X_raw, metadata = create_windows(full_df, window_size=int(target_hz * 5), overlap=0.75)
n_windows, w_size, n_channels = X_raw.shape
X_flat = X_raw.reshape(-1, n_channels)
X_final = X_flat.reshape(n_windows, w_size, n_channels)

# Step 3: Load clinical labels and build labels array
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

labels = []
for _, row in metadata.iterrows():
    ds = row["dataset"]
    sid = str(row["subjectID"]).strip()
    lbl = clinical_map.get((ds, sid), np.nan)
    labels.append(lbl)

labels = np.array(labels, dtype=np.float32)

# Filter out windows without labels
has_label_mask = ~np.isnan(labels)
num_total = len(labels)
num_labeled = int(has_label_mask.sum())
print(f"Total windows: {num_total}, labeled windows: {num_labeled}")

os.makedirs("posturalInstability/huf_clinical/data/windowed_data", exist_ok=True)

if num_labeled == 0:
    print("Warning: no labeled windows found.")
else:
    X_final = X_final[has_label_mask]
    labels = labels[has_label_mask]
    metadata = metadata.loc[has_label_mask].reset_index(drop=True)
    labels = np.clip(labels, 0, 4).astype(np.int64)

    # Step 4: Subject-level split (no window mixing per subject)
    subject_keys = metadata.apply(lambda r: (r['dataset'], str(r['subjectID']).strip()), axis=1)
    metadata = metadata.reset_index(drop=True)

    subj_to_idx = {}
    for idx, key in enumerate(subject_keys):
        subj_to_idx.setdefault(key, []).append(idx)

    unique_subjects = list(subj_to_idx.keys())

    # Compute subject-level labels for stratification
    subj_labels = []
    for key in unique_subjects:
        idxs = subj_to_idx[key]
        vals = labels[idxs]
        if len(vals) == 0:
            subj_labels.append(0)
        else:
            subj_labels.append(int(pd.Series(vals).mode().iloc[0]))

    # Split subjects with stratification fallback
    subj_label_counts = pd.Series(subj_labels).value_counts()
    if (subj_label_counts < 2).any() or len(unique_subjects) < 2:
        print("Warning: some subject-level classes have <2 members; using non-stratified split.")
        train_subj, test_subj = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    else:
        train_subj, test_subj = train_test_split(
            unique_subjects, test_size=0.2, random_state=42, stratify=subj_labels
        )

    # Further split train into train/val
    train_subj_labels = [subj_labels[unique_subjects.index(s)] for s in train_subj]
    train_label_counts = pd.Series(train_subj_labels).value_counts()
    if (train_label_counts < 2).any() or len(train_subj) < 2:
        print("Warning: train labels have <2 members; using non-stratified train/val split.")
        train_subj_final, val_subj = train_test_split(train_subj, test_size=0.125, random_state=42)
    else:
        train_subj_final, val_subj = train_test_split(
            train_subj, test_size=0.125, random_state=42, stratify=train_subj_labels
        )

    # Build indices preserving order
    train_idx = sorted([i for s in train_subj_final for i in subj_to_idx[s]])
    val_idx = sorted([i for s in val_subj for i in subj_to_idx[s]])
    test_idx = sorted([i for s in test_subj for i in subj_to_idx[s]])

    X_train = X_final[train_idx]
    y_train = labels[train_idx]
    meta_train = metadata.iloc[train_idx].reset_index(drop=True)

    X_val = X_final[val_idx]
    y_val = labels[val_idx]
    meta_val = metadata.iloc[val_idx].reset_index(drop=True)

    X_test = X_final[test_idx]
    y_test = labels[test_idx]
    meta_test = metadata.iloc[test_idx].reset_index(drop=True)

    # Step 5: Apply augmentation ONLY on training set
    aug_windows = []
    aug_labels = []
    aug_meta = []
    for ds in meta_train['dataset'].unique():
        mask = meta_train['dataset'] == ds
        if not mask.any():
            continue
        ds_w = X_train[mask.to_numpy()]
        ds_y = y_train[mask.to_numpy()]
        ds_meta = meta_train.loc[mask].reset_index(drop=True)

        bw, by, bm = oversample_dataset_subset(ds_w, ds_y, ds_meta, k_neighbors=3)
        aug_windows.append(bw)
        aug_labels.append(by)
        aug_meta.append(bm)

    if aug_windows:
        X_train = np.concatenate(aug_windows, axis=0)
        y_train = np.concatenate(aug_labels, axis=0)
        meta_train = pd.concat(aug_meta, ignore_index=True)

    # Shuffle training set after augmentation
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    meta_train = meta_train.iloc[perm].reset_index(drop=True)

    # Step 6: Save split data
    os.makedirs("posturalInstability/huf_clinical/data/windowed_data/train", exist_ok=True)
    os.makedirs("posturalInstability/huf_clinical/data/windowed_data/val", exist_ok=True)
    os.makedirs("posturalInstability/huf_clinical/data/windowed_data/test", exist_ok=True)

    np.save("posturalInstability/huf_clinical/data/windowed_data/train/windows.npy", X_train.astype(np.float32))
    np.save("posturalInstability/huf_clinical/data/windowed_data/train/labels.npy", y_train)
    meta_train.to_csv("posturalInstability/huf_clinical/data/windowed_data/train/metadata.csv", index=False)

    np.save("posturalInstability/huf_clinical/data/windowed_data/val/windows.npy", X_val.astype(np.float32))
    np.save("posturalInstability/huf_clinical/data/windowed_data/val/labels.npy", y_val)
    meta_val.to_csv("posturalInstability/huf_clinical/data/windowed_data/val/metadata.csv", index=False)

    np.save("posturalInstability/huf_clinical/data/windowed_data/test/windows.npy", X_test.astype(np.float32))
    np.save("posturalInstability/huf_clinical/data/windowed_data/test/labels.npy", y_test)
    meta_test.to_csv("posturalInstability/huf_clinical/data/windowed_data/test/metadata.csv", index=False)

    print(f"\nPreprocessing complete!")
    print(f"Train: {len(y_train)} windows ({len(np.unique(y_train))} classes)")
    print(f"Val: {len(y_val)} windows")
    print(f"Test: {len(y_test)} windows")
