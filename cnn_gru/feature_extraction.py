import argparse
import sys
import os
from model import CnnGru, ImuEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from train import train_autoencoder, train_classifier

from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split


DATA_PATH       = "posturalInstability/cnn_gru/data/"
CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH    = "posturalInstability/cnn_gru/results/"
RANDOM_STATE    = 42
FS              = 64
LATENT_DIM      = 8

def _lazy_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except ImportError as first_error:
        try:
            import umap.umap_ as umap  # type: ignore
            return umap
        except ImportError:
            raise ImportError("UMAP is not installed.") from first_error

def make_umap_plot(df, output_dir, mode_tag):
    umap = _lazy_import_umap()
    feat_cols = [c for c in df.columns if c.startswith("Feat_")]
    if not feat_cols: return
    
    # Ora il blocco latente è di 32 dimensioni (4 stats * 8 dims)
    n_latent_cols = 4 * LATENT_DIM  
    lat_cols = feat_cols[:n_latent_cols] if len(feat_cols) >= n_latent_cols else feat_cols
    latent_values = df[lat_cols].to_numpy(dtype=np.float32)
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    embedding = reducer.fit_transform(latent_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for dataset_name in pd.unique(df["dataset"]):
        mask = df["dataset"] == dataset_name
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], label=dataset_name, alpha=0.65, s=35)
    axes[0].set_title("Latent space by dataset")
    axes[0].legend(loc="best", fontsize=8)

    label_colors = {0: "steelblue", 1: "tomato"}
    label_names = {0: "HC", 1: "PD"}
    for lbl, col in label_colors.items():
        mask = df["y_true"] == lbl
        axes[1].scatter(embedding[mask, 0], embedding[mask, 1], c=col, label=label_names[lbl], alpha=0.65, s=35)
    axes[1].set_title("Latent space by class")
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

    labels_4cls   = np.clip(labels_raw, 0, 3).astype(np.int32)
    binary_labels = (labels_4cls >= 1).astype(np.float32)

    print(f"Windows        : {windows.shape}")
    print(f"4-class dist   : {dict(zip(*np.unique(labels_4cls, return_counts=True)))}")
    print(f"Binary dist    : {dict(zip(*np.unique(binary_labels, return_counts=True)))}")
    return windows, labels_4cls, binary_labels, metadata

def subject_mask(metadata, subjects_df):
    meta_idx = pd.MultiIndex.from_frame(metadata[["dataset", "subjectID"]])
    subj_idx = pd.MultiIndex.from_frame(subjects_df[["dataset", "subjectID"]])
    return meta_idx.isin(subj_idx)

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
        if len(cls) < 3:
            train_list.append(cls)
            continue
        s_trval, s_test = train_test_split(cls, test_size=0.25, random_state=RANDOM_STATE)
        s_tr, s_val     = train_test_split(s_trval, test_size=0.2, random_state=RANDOM_STATE)
        train_list.append(s_tr)
        val_list.append(s_val)
        test_list.append(s_test)

    s_train = pd.concat(train_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    s_val   = pd.concat(val_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    s_test  = pd.concat(test_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    for df in (s_train, s_val, s_test): df.drop(columns=["bin_label"], inplace=True)
    print(f"\nSplit — train: {len(s_train)}, val: {len(s_val)}, test: {len(s_test)} patients")
    return s_train, s_val, s_test

def ensure_validation_domain_coverage(s_train, s_val):
    train_domains = list(pd.unique(s_train["dataset"]))
    val_domains = set(pd.unique(s_val["dataset"]))
    missing = [ds for ds in train_domains if ds not in val_domains]
    if not missing: return s_train, s_val

    s_train = s_train.copy()
    s_val = s_val.copy()
    moved_rows = []
    for ds_name in missing:
        candidates = s_train[s_train["dataset"] == ds_name]
        if len(candidates) <= 1: continue
        row = candidates.iloc[[0]]
        s_train = s_train.drop(index=row.index)
        moved_rows.append(row)

    if moved_rows:
        s_val = pd.concat([s_val] + moved_rows, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        s_train = s_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return s_train, s_val

# ═════════════════════════════════════════════
# 2. HANDCRAFTED FEATURES
# ═════════════════════════════════════════════

def _bandpass(signal, lo=0.03, hi=1.0, fs=FS, order=2):
    nyq  = 0.5 * fs
    lo_n = np.clip(lo / nyq, 1e-5, 0.99)
    hi_n = np.clip(hi / nyq, lo_n + 1e-5, 0.99)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    if len(signal) < max(len(b), len(a)) * 3: return signal
    return filtfilt(b, a, signal)

def stance_features(window, fs=FS):
    acc_ml = _bandpass(window[:, 1], fs=fs)
    acc_ap = _bandpass(window[:, 2], fs=fs)
    cov       = np.cov(acc_ap, acc_ml)
    det       = np.linalg.det(cov)
    sway_area = np.pi * 5.991 * np.sqrt(max(det, 0.0))
    ml_var = np.var(acc_ml)
    ap_var = np.var(acc_ap)
    lateral_dominance = ml_var / (ap_var + 1e-8)
    fft_ml = np.abs(np.fft.rfft(acc_ml)) ** 2
    freqs  = np.fft.rfftfreq(len(acc_ml), d=1.0 / fs)
    p_tot  = np.sum(fft_ml) + 1e-10
    p_sway = np.sum(fft_ml[(freqs >= 0.1) & (freqs <= 0.5)]) / p_tot
    p_tremor = np.sum(fft_ml[(freqs >= 8) & (freqs <= 12)]) / p_tot
    return np.array([sway_area, lateral_dominance, p_sway, p_tremor], dtype=np.float32)

def walking_features(window, fs=FS):
    acc_v    = window[:, 0]
    acc_ap   = window[:, 2]
    jerk_mag = np.linalg.norm(np.diff(window[:, :3], axis=0) * fs, axis=1)
    norm_jerk = np.sum(jerk_mag) / (len(acc_v) / fs + 1e-8)
    peaks, _ = find_peaks(acc_v, distance=int(fs * 0.3), prominence=np.std(acc_v) * 0.3)
    step_cv = 0.0
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fs
        step_cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
    fft_ap = np.abs(np.fft.rfft(acc_ap))
    freqs  = np.fft.rfftfreq(len(acc_ap), d=1.0 / fs)
    valid  = (freqs > 0.5) & (freqs < 4.0)
    dom_freq = float(freqs[np.argmax(fft_ap[valid])]) if valid.any() else 0.0
    return np.array([norm_jerk, step_cv, dom_freq if np.isfinite(dom_freq) else 0.0], dtype=np.float32)

# ═════════════════════════════════════════════
# 3. PATIENT-LEVEL FEATURE EXTRACTION (46 DIMS)
# ═════════════════════════════════════════════

def _agg(rows, n_feat):
    if len(rows) == 0: return np.zeros(n_feat * 2, dtype=np.float32)
    arr = np.array(rows, dtype=np.float32)
    return np.concatenate([arr.mean(0), arr.std(0)])

def _lat_agg(rows):
    if len(rows) == 0: return None
    arr = np.array(rows, dtype=np.float32)  
    n = arr.shape[0]
    if n >= 3:
        t = np.arange(n, dtype=np.float32)
        t_centered = t - t.mean()
        t_var = np.dot(t_centered, t_centered) + 1e-8
        slopes = np.dot(t_centered, arr) / t_var
    else:
        slopes = np.zeros(arr.shape[1], dtype=np.float32)
    return np.concatenate([arr.mean(0), arr.std(0), arr.max(0), slopes])

def extract_patient_features(windows, binary_labels, labels_4cls, metadata, subjects_df, enc_shared):
    X, y, y_4cls, dsets, sids = [], [], [], [], []
    
    # NaN shapes: Latent (32), HC Stance (8), HC Walk (6)
    nan_lat = np.full(LATENT_DIM * 4, np.nan, dtype=np.float32)
    nan_hc_s = np.full(8, np.nan, dtype=np.float32)
    nan_hc_w = np.full(6, np.nan, dtype=np.float32)

    for _, s in subjects_df.iterrows():
        m = (metadata["dataset"] == s["dataset"]) & (metadata["subjectID"] == s["subjectID"])
        if m.sum() == 0: continue

        p_win  = windows[m].astype("float32")
        p_meta = metadata[m].reset_index(drop=True)
        p_win -= p_win.mean(axis=(0, 1), keepdims=True)
        p_win = p_win / (p_win.std(axis=(0, 1), keepdims=True) + 1e-8)

        is_stance  = ((p_meta["taskID"] == 0) | (p_meta["taskID"] == 1)).values
        is_walking = (p_meta["taskID"] == 2).values
        is_all     = is_stance | is_walking  # Tutte le finestre insieme
        
        has_s = is_stance.any()
        has_w = is_walking.any()
        has_any = is_all.any()

        # 1. LATENT GENERALI: Passo TUTTE le finestre insieme all'encoder
        lat = _lat_agg(list(enc_shared.get_latent(p_win[is_all]).numpy())) if has_any else nan_lat
        
        # 2. HANDCRAFTED SPECIFICHE: Calcolo Sway solo su stance, Jerk solo su walk
        hc_s = _agg([stance_features(w)  for w in p_win[is_stance]],  4) if has_s else nan_hc_s
        hc_w = _agg([walking_features(w) for w in p_win[is_walking]], 3) if has_w else nan_hc_w

        # Uniamo le feature (32 + 8 + 6 = 46)
        X.append(np.concatenate([lat, hc_s, hc_w]))
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

    if arch_mode == "autoencoder":
        model = ImuEncoder(input_shape=input_shape, latent_dim=LATENT_DIM)
        model(tf.zeros((1,) + input_shape)) 
        if os.path.exists(full_path):
            print(f"Loading {arch_mode} weights for {task_name}...")
            model.load_weights(full_path)
        else:
            print(f"Training {arch_mode} for {task_name}...")            
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
            train_classifier(model, x_train, y_train, x_val, y_val, CHECKPOINT_PATH, weights_filename)
    
    return model

def impute_features(train_df, test_df):
    feat_cols = [c for c in train_df.columns if c.startswith("Feat_")]
    train_group_means = {}
    global_means = train_df[feat_cols].mean(skipna=True)
    
    for label in train_df["y_true"].unique():
        mask = train_df["y_true"] == label
        grp = train_df.loc[mask, feat_cols]
        train_group_means[label] = grp.mean(skipna=True).fillna(global_means)

    train_df_imp = train_df.copy()
    test_df_imp = test_df.copy()
    
    for label, mean_vals in train_group_means.items():
        mask_tr = train_df_imp["y_true"] == label
        train_df_imp.loc[mask_tr, feat_cols] = train_df_imp.loc[mask_tr, feat_cols].fillna(mean_vals)
        
        mask_te = test_df_imp["y_true"] == label
        test_df_imp.loc[mask_te, feat_cols] = test_df_imp.loc[mask_te, feat_cols].fillna(mean_vals)

    train_df_imp[feat_cols] = train_df_imp[feat_cols].fillna(global_means)
    test_df_imp[feat_cols] = test_df_imp[feat_cols].fillna(global_means)
    return train_df_imp, test_df_imp

def main(arch_mode="autoencoder"):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    print("Loading datasets...")
    windows, labels_4cls, binary_labels, metadata = load_data()
    input_shape = windows.shape[1:]

    s_train, s_val, s_test = patient_split(metadata, binary_labels)
    s_train, s_val = ensure_validation_domain_coverage(s_train, s_val)

    enc_shared = get_or_train_encoder(
        task_name="shared", windows=windows, labels_4cls=labels_4cls, metadata=metadata, 
        s_train=s_train, s_val=s_val, task_filter_fn=lambda m: (m["taskID"] <= 2),
        input_shape=input_shape, arch_mode=arch_mode
    )

    s_fit = pd.concat([s_train, s_val]).reset_index(drop=True)
    print("\nExtracting patient-level features...")
    X_fit, y_fit, y4_fit, d_fit, s_fit_ids = extract_patient_features(windows, binary_labels, labels_4cls, metadata, s_fit, enc_shared)
    X_test, y_test, y4_test, d_test, s_test_ids = extract_patient_features(windows, binary_labels, labels_4cls, metadata, s_test, enc_shared)

    print(f"Full stance+walking feature vector : {X_fit.shape[1]} dims")

    train_df = pd.DataFrame(X_fit, columns=[f"Feat_{i}" for i in range(X_fit.shape[1])])
    train_df["y_true"] = y_fit
    train_df["y_true_4cls"] = y4_fit
    train_df["dataset"] = d_fit
    train_df["subjectID"] = s_fit_ids

    test_df = pd.DataFrame(X_test, columns=[f"Feat_{i}" for i in range(X_test.shape[1])])
    test_df["y_true"] = y_test
    test_df["y_true_4cls"] = y4_test
    test_df["dataset"] = d_test
    test_df["subjectID"] = s_test_ids

    train_df, test_df = impute_features(train_df, test_df)

    train_out = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{arch_mode}.csv")
    test_out  = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{arch_mode}.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Saved -> {train_out}")
    print(f"Saved -> {test_out}")

    try:
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        umap_path = make_umap_plot(full_df, RESULTS_PATH, arch_mode)
        print(f"UMAP saved -> {umap_path}")
    except Exception as e:
        print(f"Warning: failed to create UMAP: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument("--arch-mode", choices=["autoencoder", "classifier"], default="autoencoder")
    args = parser.parse_args()
    main(arch_mode=args.arch_mode)