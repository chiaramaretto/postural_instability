import tensorflow as tf
from model import CnnGru, ImuEncoder
from train import train, train_autoencoder
from train import train_autoencoder_mmd
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import iqr
from sklearn.model_selection import train_test_split


DATA_PATH       = "posturalInstability/cnn_gru/data/"
CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH    = "posturalInstability/cnn_gru/results/"
RANDOM_STATE    = 42
FS              = 128.0
LATENT_DIM      = 16  
ARCH_MODE       = "autoencoder"  # "autoencoder" or "classifier"
USE_MMD         = True
TARGET_DATASET  = None
LAMBDA_MMD      = 0.1


# ═════════════════════════════════════════════
# 1. DATA LOADING AND SPLITTING
# ═════════════════════════════════════════════

def load_data():
    windows      = np.load(os.path.join(DATA_PATH, "windows.npy"))
    labels_raw   = np.load(os.path.join(DATA_PATH, "labels.npy")).astype(np.float32)
    metadata     = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))

    labels_4cls  = np.clip(labels_raw, 0, 3).astype(np.int32) 
    binary_labels = (labels_4cls >= 1).astype(np.float32)

    print(f"Windows        : {windows.shape}")
    print(f"4-class dist   : {dict(zip(*np.unique(labels_4cls, return_counts=True)))}")
    print(f"Binary dist    : {dict(zip(*np.unique(binary_labels, return_counts=True)))}")
    return windows, labels_4cls, binary_labels, metadata


def subject_mask(metadata, subjects_df):
    meta_idx = pd.MultiIndex.from_frame(metadata[["dataset", "subjectID"]])
    subj_idx = pd.MultiIndex.from_frame(subjects_df[["dataset", "subjectID"]])
    return meta_idx.isin(subj_idx)


def split_windows_by_dataset(windows, labels_4cls, metadata, mask):
    groups_x, groups_y, groups_name = [], [], []
    datasets = metadata.loc[mask, "dataset"].drop_duplicates().tolist()

    for ds_name in datasets:
        ds_mask = mask & (metadata["dataset"] == ds_name)
        x_group = windows[ds_mask]
        y_group = labels_4cls[ds_mask]
        if len(x_group) == 0:
            continue
        groups_x.append(x_group)
        groups_y.append(y_group)
        groups_name.append(ds_name)

    return groups_x, groups_y, groups_name


def patient_split(metadata, binary_labels):
    subjects = metadata[["dataset", "subjectID"]].drop_duplicates().copy()

    subj_bin = []
    for _, s in subjects.iterrows():
        m = (metadata["dataset"] == s["dataset"]) & (metadata["subjectID"] == s["subjectID"])
        subj_bin.append(int(binary_labels[m][0]))
    subjects["bin_label"] = subj_bin

    train_list, val_list, test_list = [], [], []
    for lbl in [0, 1]:
        cls = subjects[subjects["bin_label"] == lbl]
        print(f"Class {lbl} patients: {len(cls)}")
        if len(cls) < 3:
            train_list.append(cls)
            continue
    
        s_trval, s_test = train_test_split(cls, test_size=0.25, random_state=RANDOM_STATE)
        s_tr, s_val     = train_test_split(s_trval, test_size=0.3, random_state=RANDOM_STATE)
        train_list.append(s_tr)
        val_list.append(s_val)
        test_list.append(s_test)

    s_train = pd.concat(train_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    s_val   = pd.concat(val_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    s_test  = pd.concat(test_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    for df in (s_train, s_val, s_test):
        df.drop(columns=["bin_label"], inplace=True)

    print(f"\nSplit — train: {len(s_train)}, val: {len(s_val)}, test: {len(s_test)} patients")
    return s_train, s_val, s_test


def ensure_validation_domain_coverage(s_train, s_val):
    train_domains = list(pd.unique(s_train["dataset"]))
    val_domains = set(pd.unique(s_val["dataset"]))
    missing = [ds for ds in train_domains if ds not in val_domains]

    if not missing:
        return s_train, s_val

    s_train = s_train.copy()
    s_val = s_val.copy()
    moved_rows = []

    for ds_name in missing:
        candidates = s_train[s_train["dataset"] == ds_name]
        if len(candidates) <= 1:
            print(f"WARNING: cannot move a subject from domain {ds_name} into validation without emptying training.")
            continue

        row = candidates.iloc[[0]]
        s_train = s_train.drop(index=row.index)
        moved_rows.append(row)

    if moved_rows:
        s_val = pd.concat([s_val] + moved_rows, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        s_train = s_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    remaining_missing = [ds for ds in pd.unique(s_train["dataset"]) if ds not in set(pd.unique(s_val["dataset"]))]
    if remaining_missing:
        print(f"WARNING: validation still misses domains: {remaining_missing}")

    return s_train, s_val


# ═════════════════════════════════════════════
# 2. HANDCRAFTED FEATURES
# ═════════════════════════════════════════════

def _bandpass(signal, lo=0.03, hi=1.0, fs=FS, order=2):
    nyq  = 0.5 * fs
    lo_n = np.clip(lo / nyq, 1e-5, 0.99)
    hi_n = np.clip(hi / nyq, lo_n + 1e-5, 0.99)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    if len(signal) < max(len(b), len(a)) * 3:
        return signal
    return filtfilt(b, a, signal)

def stance_features(window, fs=FS):
    # Mapping: x(0)=Vertical, y(1)=Medio-lateral, z(2)=Antero-frontal
    acc_v  = window[:, 0]
    acc_ml = _bandpass(window[:, 1], fs=fs)
    acc_ap = _bandpass(window[:, 2], fs=fs)

    cov       = np.cov(acc_ap, acc_ml)
    det       = np.linalg.det(cov)
    sway_area = np.pi * 5.991 * np.sqrt(max(det, 0.0))

    ml_std      = np.std(acc_ml)
    ap_std      = np.std(acc_ap)
    ml_ap_ratio = ml_std / (ap_std + 1e-8)
    mean_cop_ap = np.mean(acc_ap)

    fft_ml = np.abs(np.fft.rfft(acc_ml)) ** 2
    freqs  = np.fft.rfftfreq(len(acc_ml), d=1.0 / fs)
    p_tot  = np.sum(fft_ml) + 1e-10
    p_8_12 = np.sum(fft_ml[(freqs >= 8)   & (freqs <= 12)]) / p_tot
    p_sway = np.sum(fft_ml[(freqs >= 0.1) & (freqs <= 0.5)]) / p_tot

    rms_v       = np.sqrt(np.mean(acc_v ** 2))
    iqr_v       = iqr(acc_v)
    gyro_ml_std = np.std(window[:, 4])

    return np.array([
        sway_area, ml_std, ap_std, ml_ap_ratio, mean_cop_ap,
        p_8_12, p_sway, rms_v, iqr_v, gyro_ml_std,
        np.mean(np.abs(acc_ml)),
        np.max(np.abs(acc_ml)),
    ], dtype=np.float32)

def walking_features(window, fs=FS):
    acc_v    = window[:, 0]
    acc_ml   = window[:, 1]
    acc_ap   = window[:, 2]
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)

    peaks, _ = find_peaks(acc_v, distance=int(fs * 0.3),
                          prominence=np.std(acc_v) * 0.3)
    n_steps  = len(peaks)
    step_cv  = 0.0
    if n_steps > 1:
        intervals = np.diff(peaks) / fs
        step_cv   = np.std(intervals) / (np.mean(intervals) + 1e-8)

    jerk_mag  = np.linalg.norm(np.diff(window[:, :3], axis=0) * fs, axis=1)
    norm_jerk = np.sum(jerk_mag) / (len(acc_v) / fs + 1e-8)

    fft_ap = np.abs(np.fft.rfft(acc_ap))
    freqs  = np.fft.rfftfreq(len(acc_ap), d=1.0 / fs)
    valid  = (freqs > 0.5) & (freqs < 4.0)
    dom_freq = float(freqs[np.argmax(fft_ap[valid])]) if valid.any() else 0.0

    return np.array([
        norm_jerk, np.mean(jerk_mag), np.std(jerk_mag),
        step_cv, float(n_steps),
        np.std(acc_ml), np.mean(gyro_mag), np.std(gyro_mag),
        np.std(acc_v), dom_freq if np.isfinite(dom_freq) else 0.0,
    ], dtype=np.float32)

def turning_features(window, fs=FS):
    gyro_v   = window[:, 3] # Yaw rotation around vertical axis (0) corresponds to gyro x (3)
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)
    acc_mag  = np.linalg.norm(window[:, :3], axis=1)
    peaks, _ = find_peaks(acc_mag, distance=20, prominence=np.std(acc_mag) * 0.3)
    return np.array([
        np.mean(np.abs(gyro_mag)), np.std(gyro_mag),
        np.max(np.abs(gyro_v)), float(len(peaks)),
        np.std(acc_mag), np.mean(np.abs(np.diff(gyro_v))),
    ], dtype=np.float32)


# ═════════════════════════════════════════════
# 3. PATIENT-LEVEL FEATURE EXTRACTION
# ═════════════════════════════════════════════

def _agg(rows, n_feat):
    if len(rows) == 0:
        return np.zeros(n_feat * 2, dtype=np.float32)
    arr = np.array(rows, dtype=np.float32)
    return np.concatenate([arr.mean(0), arr.std(0)])

def _lat_agg(rows):
    if len(rows) == 0:
        return None
    arr = np.array(rows, dtype=np.float32)
    return np.concatenate([arr.mean(0), arr.std(0), arr.max(0)])

def extract_patient_features(windows, binary_labels, labels_4cls, metadata,
                              subjects_df, enc_stance, enc_walk):
    X, y, y_4cls, dsets, sids = [], [], [], [], []
    zero_lat = np.zeros(LATENT_DIM * 3, dtype=np.float32)

    for _, s in subjects_df.iterrows():
        m = (metadata["dataset"] == s["dataset"]) & \
            (metadata["subjectID"] == s["subjectID"])
        if m.sum() == 0:
            continue

        p_win  = windows[m].astype("float32")
        p_meta = metadata[m].reset_index(drop=True)
        p_win -= p_win.mean(axis=(0, 1), keepdims=True)

        is_stance  = ((p_meta["taskID"] == 0) | (p_meta["taskID"] == 1)).values
        is_walking = ((p_meta["taskID"] == 2) & (p_meta["isTurn"] == 0)).values
        is_turning = ((p_meta["taskID"] == 2) & (p_meta["isTurn"] == 1)).values

        has_s = is_stance.any()
        has_w = is_walking.any()
        has_t = is_turning.any()

        lat_s = _lat_agg(list(enc_stance.get_latent(p_win[is_stance]).numpy())) if has_s else None
        lat_w = _lat_agg(list(enc_walk.get_latent(p_win[is_walking]).numpy())) if has_w else None
        lat_t = _lat_agg(list(enc_walk.get_latent(p_win[is_turning]).numpy())) if has_t else None

        lat_vec = np.concatenate([
            [float(has_s)], lat_s if lat_s is not None else zero_lat,
            [float(has_w)], lat_w if lat_w is not None else zero_lat,
            [float(has_t)], lat_t if lat_t is not None else zero_lat,
        ])

        hc_s = _agg([stance_features(w)  for w in p_win[is_stance]],  12) if has_s else np.zeros(24, np.float32)
        hc_w = _agg([walking_features(w) for w in p_win[is_walking]], 10) if has_w else np.zeros(20, np.float32)
        hc_t = _agg([turning_features(w) for w in p_win[is_turning]], 6)  if has_t else np.zeros(12, np.float32)

        hc_vec = np.concatenate([[float(has_s), float(has_w), float(has_t)], hc_s, hc_w, hc_t])

        X.append(np.concatenate([lat_vec, hc_vec]))
        y.append(float(binary_labels[m][0]))
        y_4cls.append(int(labels_4cls[m][0]))
        dsets.append(s["dataset"])
        sids.append(s["subjectID"])

    return np.stack(X), np.array(y), np.array(y_4cls), np.array(dsets), np.array(sids)

# ═════════════════════════════════════════════
# 4. ENCODER TRAINING
# ═════════════════════════════════════════════

def get_or_train_encoder(task_name, windows, labels_4cls, metadata, s_train, s_val, 
                         task_filter_fn, input_shape, arch_mode):
    train_mask = subject_mask(metadata, s_train) & task_filter_fn(metadata)
    val_mask   = subject_mask(metadata, s_val)   & task_filter_fn(metadata)
    
    x_train = windows[train_mask]
    x_val   = windows[val_mask]

    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        raise ValueError(f"No windows available for {task_name} encoder.")
    
    weights_filename = f"best_{arch_mode}_{task_name}.weights.h5"
    full_path = os.path.join(CHECKPOINT_PATH, weights_filename)

    # If MMD is enabled, prefer a separate weights filename with _mmd suffix
    if USE_MMD:
        suffix = f"_mmd_{TARGET_DATASET}" if TARGET_DATASET else "_mmd"
        weights_mmd = weights_filename.replace('.weights.h5', f'{suffix}.weights.h5')
        weights_filename_mmd = weights_mmd
        full_path = os.path.join(CHECKPOINT_PATH, weights_filename_mmd)

    if arch_mode == "autoencoder":
        model = ImuEncoder(input_shape=input_shape)
        model(tf.zeros((1,) + input_shape)) 
        if os.path.exists(full_path):
            print(f"Loading {arch_mode} weights for {task_name}...")
            model.load_weights(full_path)
        else:
            print(f"Training {arch_mode} for {task_name}...")
            if USE_MMD:
                train_groups, _, _ = split_windows_by_dataset(windows, labels_4cls, metadata, train_mask)
                val_groups, _, _ = split_windows_by_dataset(windows, labels_4cls, metadata, val_mask)

                if len(train_groups) < 2:
                    raise ValueError(f"Need at least two dataset groups for MMD training in {task_name} encoder.")

                train_autoencoder_mmd(
                    model, train_groups, None, val_groups if len(val_groups) > 0 else None, None,
                    CHECKPOINT_PATH, weights_filename=weights_filename_mmd,
                    lambda_mmd=LAMBDA_MMD
                )
            else:
                train_autoencoder(model, x_train, x_val, CHECKPOINT_PATH, weights_filename)

    elif arch_mode == "classifier":
        y_train = labels_4cls[train_mask]
        y_val   = labels_4cls[val_mask]
        n_classes = len(np.unique(labels_4cls)) if len(labels_4cls) > 0 else 4
        
        model = CnnGru(input_shape=input_shape, n_classes=n_classes)
        model(tf.zeros((1,) + input_shape)) 
        
        if os.path.exists(full_path):
            print(f"Loading {arch_mode} weights for {task_name}...")
            model.load_weights(full_path)
        else:
            print(f"Training {arch_mode} for {task_name}...")
            if USE_MMD:
                train_groups, label_groups, _ = split_windows_by_dataset(windows, labels_4cls, metadata, train_mask)
                val_groups, val_label_groups, _ = split_windows_by_dataset(windows, labels_4cls, metadata, val_mask)

                if len(train_groups) < 2:
                    raise ValueError(f"Need at least two dataset groups for MMD training in {task_name} encoder.")

                from train import train_classifier_mmd
                train_classifier_mmd(
                    model, train_groups, label_groups, None,
                    x_val_source=val_groups if len(val_groups) > 0 else None,
                    y_val_source=val_label_groups if len(val_label_groups) > 0 else None,
                    checkpoint_path=CHECKPOINT_PATH, weights_filename=weights_filename_mmd,
                    lambda_mmd=LAMBDA_MMD
                )
            else:
                train(model, x_train, y_train, x_val, y_val, CHECKPOINT_PATH, weights_filename)
    
    return model

# ═════════════════════════════════════════════
# 5. MAIN (FEATURE EXTRACTION ONLY)
# ═════════════════════════════════════════════

def main():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    print("Loading datasets...")
    windows, labels_4cls, binary_labels, metadata = load_data()
    input_shape = windows.shape[1:]

    s_train, s_val, s_test = patient_split(metadata, binary_labels)
    s_train, s_val = ensure_validation_domain_coverage(s_train, s_val)

    enc_stance = get_or_train_encoder(
        task_name="stance", windows=windows, labels_4cls=labels_4cls, metadata=metadata, 
        s_train=s_train, s_val=s_val, task_filter_fn=lambda m: (m["taskID"] < 2),
        input_shape=input_shape, arch_mode=ARCH_MODE
    )
    
    enc_walk = get_or_train_encoder(
        task_name="walk", windows=windows, labels_4cls=labels_4cls, metadata=metadata, 
        s_train=s_train, s_val=s_val, task_filter_fn=lambda m: (m["taskID"] == 2),
        input_shape=input_shape, arch_mode=ARCH_MODE
    )

    print("\nExtracting patient-level features...")
    s_fit = pd.concat([s_train, s_val]).reset_index(drop=True)

    X_fit,  y_fit, y_fit_4cls, dsets_fit,  sids_fit  = extract_patient_features(windows, binary_labels, labels_4cls, metadata, s_fit,  enc_stance, enc_walk)
    X_test, y_test, y_test_4cls, dsets_test, sids_test = extract_patient_features(windows, binary_labels, labels_4cls, metadata, s_test, enc_stance, enc_walk)

    print(f"Full feature vector : {X_fit.shape[1]} dims")
    print(f"Train patients      : {len(y_fit)}")
    print(f"Test patients       : {len(y_test)}")

    # Bundle target values, subject IDs, and dataset names directly into the feature CSVs 
    # to make downstream classification clean and independent.
    train_df = pd.DataFrame(X_fit, columns=[f"Feat_{i}" for i in range(X_fit.shape[1])])
    train_df["y_true"] = y_fit
    train_df["y_true_4cls"] = y_fit_4cls
    train_df["dataset"] = dsets_fit
    train_df["subjectID"] = sids_fit

    test_df = pd.DataFrame(X_test, columns=[f"Feat_{i}" for i in range(X_test.shape[1])])
    test_df["y_true"] = y_test
    test_df["y_true_4cls"] = y_test_4cls
    test_df["dataset"] = dsets_test
    test_df["subjectID"] = sids_test

    train_out = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{ARCH_MODE}.csv")
    test_out = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{ARCH_MODE}.csv")
    if USE_MMD:
        suffix = f"_mmd_{TARGET_DATASET}" if TARGET_DATASET else "_mmd"
        train_out = train_out.replace('.csv', f'{suffix}.csv')
        test_out = test_out.replace('.csv', f'{suffix}.csv')
    
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    
    print("\nFeature extraction complete.")
    print(f"Saved -> {train_out}")
    print(f"Saved -> {test_out}")
    print("Ready to run classifier.py for the double ablation study.")

if __name__ == "__main__":
    main()