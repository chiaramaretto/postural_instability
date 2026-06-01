import argparse
from model import CnnGru, ImuEncoder
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from train import train, train_autoencoder
from train import train_autoencoder_mmd

from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split


DATA_PATH       = "posturalInstability/cnn_gru/data/"
CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH    = "posturalInstability/cnn_gru/results/"
RANDOM_STATE    = 42
FS              = 64
LATENT_DIM      = 8  
ARCH_MODE       = "classifier"  # "autoencoder" or "classifier"
USE_MMD        = False
TARGET_DATASET  = None
LAMBDA_MMD      = 0.01
LAMBDA_VALUES   = [0.001, 0.005, 0.01]


def _lazy_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except ImportError as first_error:
        try:
            import umap.umap_ as umap  # type: ignore
            return umap
        except ImportError:
            raise ImportError(
                "UMAP is not installed. Install `umap-learn` in the active environment to run this analysis."
            ) from first_error


def _format_lambda_tag(lambda_mmd):
    return f"lambda_{lambda_mmd:.3f}".replace(".", "p")


def build_mmd_suffix(lambda_mmd=None):
    lambda_value = LAMBDA_MMD if lambda_mmd is None else lambda_mmd
    suffix_parts = ["_mmd"]
    if TARGET_DATASET:
        suffix_parts.append(str(TARGET_DATASET))
    suffix_parts.append(_format_lambda_tag(lambda_value))
    return "_".join(suffix_parts)


def make_umap_plot(df, output_dir, mode_tag):
    umap = _lazy_import_umap()
    feat_cols = [c for c in df.columns if c.startswith("Feat_")]
    if not feat_cols:
        raise ValueError("No Feat_* columns available for UMAP plotting.")

    latent_values = df[feat_cols].to_numpy(dtype=np.float32)
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    embedding = reducer.fit_transform(latent_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for dataset_name in pd.unique(df["dataset"]):
        mask = df["dataset"] == dataset_name
        axes[0].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=dataset_name,
            alpha=0.65,
            s=35,
        )
    axes[0].set_title("Latent space by dataset")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend(loc="best", fontsize=8)

    label_colors = {0: "steelblue", 1: "tomato"}
    label_names = {0: "HC", 1: "PD"}
    for lbl, col in label_colors.items():
        mask = df["y_true"] == lbl
        axes[1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=col,
            label=label_names[lbl],
            alpha=0.65,
            s=35,
        )
    axes[1].set_title("Latent space by class")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].legend(loc="best")

    fig.suptitle(f"UMAP latent space - {mode_tag}", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"umap_latent_{mode_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
    acc_ml = _bandpass(window[:, 1], fs=fs)
    acc_ap = _bandpass(window[:, 2], fs=fs)

    # Sway Area
    cov       = np.cov(acc_ap, acc_ml)
    det       = np.linalg.det(cov)
    sway_area = np.pi * 5.991 * np.sqrt(max(det, 0.0))

    # Lateral Dominance
    ml_var = np.var(acc_ml)
    ap_var = np.var(acc_ap)
    lateral_dominance = ml_var / (ap_var + 1e-8)

    # Spectral Power
    fft_ml = np.abs(np.fft.rfft(acc_ml)) ** 2
    freqs  = np.fft.rfftfreq(len(acc_ml), d=1.0 / fs)
    p_tot  = np.sum(fft_ml) + 1e-10
    p_sway = np.sum(fft_ml[(freqs >= 0.1) & (freqs <= 0.5)]) / p_tot
    p_tremor = np.sum(fft_ml[(freqs >= 8) & (freqs <= 12)]) / p_tot

    return np.array([
        sway_area, lateral_dominance, p_sway, p_tremor,
    ], dtype=np.float32)

def walking_features(window, fs=FS):
    acc_v    = window[:, 0]
    acc_ap   = window[:, 2]

    # Jerk Magnitude
    jerk_mag = np.linalg.norm(np.diff(window[:, :3], axis=0) * fs, axis=1)
    norm_jerk = np.sum(jerk_mag) / (len(acc_v) / fs + 1e-8)

    # Step CV
    peaks, _ = find_peaks(acc_v, distance=int(fs * 0.3),
                          prominence=np.std(acc_v) * 0.3)
    step_cv = 0.0
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fs
        step_cv = np.std(intervals) / (np.mean(intervals) + 1e-8)

    # Dominant Frequency
    fft_ap = np.abs(np.fft.rfft(acc_ap))
    freqs  = np.fft.rfftfreq(len(acc_ap), d=1.0 / fs)
    valid  = (freqs > 0.5) & (freqs < 4.0)
    dom_freq = float(freqs[np.argmax(fft_ap[valid])]) if valid.any() else 0.0

    return np.array([
        norm_jerk, step_cv, dom_freq if np.isfinite(dom_freq) else 0.0,
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
    arr = np.array(rows, dtype=np.float32)  # shape: (n_windows, latent_dim)

    # Trend: slope of a linear regression for each latent dimension.
    # Captures whether each dimension increases, decreases, or stays stable over time.
    n = arr.shape[0]
    if n >= 3:
        t = np.arange(n, dtype=np.float32)
        t_centered = t - t.mean()
        t_var = np.dot(t_centered, t_centered) + 1e-8
        slopes = np.dot(t_centered, arr) / t_var
    else:
        slopes = np.zeros(arr.shape[1], dtype=np.float32)

    return np.concatenate([arr.mean(0), arr.std(0), arr.max(0), slopes])


def export_latent_trajectories(windows, metadata, subjects_df, enc_stance, enc_walk, output_tag):
    records = []

    for _, s in subjects_df.iterrows():
        m = (metadata["dataset"] == s["dataset"]) & (metadata["subjectID"] == s["subjectID"])
        if m.sum() == 0:
            continue

        p_win = windows[m].astype("float32")
        p_meta = metadata[m].reset_index(drop=True)
        p_win -= p_win.mean(axis=(0, 1), keepdims=True)
        p_win = p_win / (p_win.std(axis=(0, 1), keepdims=True) + 1e-8)

        sort_idx = p_meta["window_id"].argsort().values if "window_id" in p_meta.columns else np.arange(len(p_meta))
        p_win = p_win[sort_idx]
        p_meta = p_meta.iloc[sort_idx].reset_index(drop=True)

        tasks = [
            ("stance", enc_stance, (p_meta["taskID"] < 2).values),
            ("walk", enc_walk, ((p_meta["taskID"] == 2) & (p_meta["isTurn"] == 0)).values),
        ]

        for task_name, encoder, mask in tasks:
            if encoder is None or not mask.any():
                continue

            latents = encoder.get_latent(p_win[mask]).numpy()
            task_meta = p_meta.loc[mask].reset_index(drop=True)

            for order, (_, row) in enumerate(task_meta.iterrows()):
                rec = {
                    "dataset": s["dataset"],
                    "subjectID": s["subjectID"],
                    "task": task_name,
                    "window_order": order,
                    "window_id": int(row["window_id"]) if "window_id" in row and pd.notna(row["window_id"]) else order,
                    "taskID": int(row["taskID"]),
                    "isTurn": int(row["isTurn"]) if "isTurn" in row and pd.notna(row["isTurn"]) else 0,
                }
                for d in range(latents.shape[1]):
                    rec[f"lat_{d}"] = float(latents[order, d])
                records.append(rec)

    latent_df = pd.DataFrame(records)
    latent_path = os.path.join(CHECKPOINT_PATH, f"latent_trajectories_{ARCH_MODE}{output_tag}.csv")
    latent_df.to_csv(latent_path, index=False)
    print(f"Latent trajectories saved -> {latent_path}")
    return latent_df, latent_path

def extract_patient_features(windows, binary_labels, labels_4cls, metadata,
                              subjects_df, enc_stance, enc_walk=None):
    X, y, y_4cls, dsets, sids = [], [], [], [], []
    nan_lat = np.full(LATENT_DIM * 4, np.nan, dtype=np.float32)
    nan_hc_s = np.full(8, np.nan, dtype=np.float32)
    nan_hc_w = np.full(6, np.nan, dtype=np.float32)

    for _, s in subjects_df.iterrows():
        m = (metadata["dataset"] == s["dataset"]) & \
            (metadata["subjectID"] == s["subjectID"])
        if m.sum() == 0:
            continue

        p_win  = windows[m].astype("float32")
        p_meta = metadata[m].reset_index(drop=True)
        p_win -= p_win.mean(axis=(0, 1), keepdims=True)
        p_win = p_win / (p_win.std(axis=(0, 1), keepdims=True) + 1e-8)

        is_stance  = ((p_meta["taskID"] == 0) | (p_meta["taskID"] == 1)).values
        is_walking = ((p_meta["taskID"] == 2) & (p_meta["isTurn"] == 0)).values
        is_turning = ((p_meta["taskID"] == 2) & (p_meta["isTurn"] == 1)).values

        has_s = is_stance.any()
        has_w = is_walking.any()

        lat_s = _lat_agg(list(enc_stance.get_latent(p_win[is_stance]).numpy())) if has_s else nan_lat
        lat_w = _lat_agg(list(enc_walk.get_latent(p_win[is_walking]).numpy())) if (has_w and enc_walk is not None) else nan_lat

        # Turning is kept commented for later use.
        # is_turning = ((p_meta["taskID"] == 2) & (p_meta["isTurn"] == 1)).values

        lat_vec = np.concatenate([lat_s, lat_w])

        hc_s = _agg([stance_features(w)  for w in p_win[is_stance]],  4) if has_s else nan_hc_s
        hc_w = _agg([walking_features(w) for w in p_win[is_walking]], 3) if (has_w and enc_walk is not None) else nan_hc_w

        hc_vec = np.concatenate([hc_s, hc_w])

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
        suffix = build_mmd_suffix()
        weights_mmd = weights_filename.replace('.weights.h5', f'{suffix}.weights.h5')
        weights_filename_mmd = weights_mmd
        full_path = os.path.join(CHECKPOINT_PATH, weights_filename_mmd)

    if arch_mode == "autoencoder":
        model = ImuEncoder(input_shape=input_shape, latent_dim=LATENT_DIM)
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

    print(f"Full stance+walking feature vector : {X_fit.shape[1]} dims")
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

    # -------------------------
    # Group-wise mean imputation for all Feat_* columns (fit on training only)
    # -------------------------
    feat_cols = [c for c in train_df.columns if c.startswith("Feat_")]
    print(f"Imputing missing features for {len(feat_cols)} Feat_* columns using training-group means.")

    # Compute per-label means on the training set (ignore NaNs)
    train_group_means = {}
    global_means = train_df[feat_cols].mean(skipna=True)
    for label in train_df["y_true"].unique():
        mask = train_df["y_true"] == label
        # compute column-wise mean for this label (skip NaNs)
        grp = train_df.loc[mask, feat_cols]
        grp_mean = grp.mean(skipna=True)
        # fallback to global mean for any columns that remain NaN in grp_mean
        grp_mean_filled = grp_mean.fillna(global_means)
        train_group_means[label] = grp_mean_filled

    # Impute training set (fill NaNs using its group's means)
    train_df_imputed = train_df.copy()
    n_before = train_df_imputed[feat_cols].isna().sum().sum()
    for label, mean_vals in train_group_means.items():
        mask = train_df_imputed["y_true"] == label
        train_df_imputed.loc[mask, feat_cols] = train_df_imputed.loc[mask, feat_cols].fillna(mean_vals)
    n_after = train_df_imputed[feat_cols].isna().sum().sum()
    print(f"Training NaNs before: {n_before}, after imputation: {n_after}")

    # Ensure no remaining NaNs in training (fill any remaining with global means)
    train_df_imputed[feat_cols] = train_df_imputed[feat_cols].fillna(global_means)

    # Apply the same train-group means to test set (no fitting on test)
    test_df_imputed = test_df.copy()
    n_before_test = test_df_imputed[feat_cols].isna().sum().sum()
    for label, mean_vals in train_group_means.items():
        mask = test_df_imputed["y_true"] == label
        test_df_imputed.loc[mask, feat_cols] = test_df_imputed.loc[mask, feat_cols].fillna(mean_vals)
    # fallback global mean
    test_df_imputed[feat_cols] = test_df_imputed[feat_cols].fillna(global_means)
    n_after_test = test_df_imputed[feat_cols].isna().sum().sum()
    print(f"Test NaNs before: {n_before_test}, after imputation: {n_after_test}")

    train_df = train_df_imputed
    test_df = test_df_imputed

    train_out = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{ARCH_MODE}.csv")
    test_out = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{ARCH_MODE}.csv")
    if USE_MMD:
        suffix = build_mmd_suffix()
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