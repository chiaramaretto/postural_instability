from model import CnnGru, ImuEncoder
from train import train, train_autoencoder
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import iqr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, confusion_matrix, roc_auc_score, 
    precision_score, recall_score
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split



DATA_PATH       = "posturalInstability/cnn_gru/data/"
CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH    = "posturalInstability/cnn_gru/results/"
RANDOM_STATE    = 42
FS              = 128.0
LATENT_DIM      = 16  
ARCH_MODE       = "autoencoder"  # "autoencoder" or "classifier"


# ═════════════════════════════════════════════
# 1. DATA LOADING AND SPLITTING
# ═════════════════════════════════════════════

def load_data():
    windows      = np.load(os.path.join(DATA_PATH, "windows.npy"))
    labels_raw   = np.load(os.path.join(DATA_PATH, "labels.npy")).astype(np.float32)
    metadata     = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))

    # Clip label 4 → 3 (only 1 subject in wearpd)
    labels_4cls  = np.clip(labels_raw, 0, 3).astype(np.int32)   # 4-class for pre-training

    # Binary target for final classification
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
    """Stratified split by patient keeping windows together."""
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



# ═════════════════════════════════════════════
# 3. HANDCRAFTED FEATURES
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
    """
    12 features inspired by Biaszczyk 2007 (sway) and Smith 2025 (spectral).
    Channels: acc_x(AP), acc_y(ML), acc_z(V), gyro_x, gyro_y, gyro_z
    """
    acc_ap = _bandpass(window[:, 0], fs=fs)
    acc_ml = _bandpass(window[:, 1], fs=fs)
    acc_v  = window[:, 2]

    cov       = np.cov(acc_ap, acc_ml)
    det       = np.linalg.det(cov)
    sway_area = np.pi * 5.991 * np.sqrt(max(det, 0.0))  # 95% ellipse (Biaszczyk)

    ml_std      = np.std(acc_ml)
    ap_std      = np.std(acc_ap)
    ml_ap_ratio = ml_std / (ap_std + 1e-8)   # lateral dominance
    mean_cop_ap = np.mean(acc_ap)             # forward lean

    fft_ml = np.abs(np.fft.rfft(acc_ml)) ** 2
    freqs  = np.fft.rfftfreq(len(acc_ml), d=1.0 / fs)
    p_tot  = np.sum(fft_ml) + 1e-10
    p_8_12 = np.sum(fft_ml[(freqs >= 8)   & (freqs <= 12)]) / p_tot  # Smith 2025
    p_sway = np.sum(fft_ml[(freqs >= 0.1) & (freqs <= 0.5)]) / p_tot

    rms_v       = np.sqrt(np.mean(acc_v ** 2))
    iqr_v       = iqr(acc_v)
    gyro_ml_std = np.std(window[:, 4])

    return np.array([
        sway_area, ml_std, ap_std, ml_ap_ratio, mean_cop_ap,
        p_8_12, p_sway, rms_v, iqr_v, gyro_ml_std,
        np.mean(np.abs(acc_ml)),
        np.max(np.abs(acc_ml)),
    ], dtype=np.float32)   # 12 features


def walking_features(window, fs=FS):
    """10 features for straight walking windows."""
    acc_ap   = window[:, 0]
    acc_ml   = window[:, 1]
    acc_v    = window[:, 2]
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
    ], dtype=np.float32)   # 10 features


def turning_features(window, fs=FS):
    """6 features for turning windows."""
    gyro_z   = window[:, 5]
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)
    acc_mag  = np.linalg.norm(window[:, :3], axis=1)
    peaks, _ = find_peaks(acc_mag, distance=20, prominence=np.std(acc_mag) * 0.3)
    return np.array([
        np.mean(np.abs(gyro_mag)), np.std(gyro_mag),
        np.max(np.abs(gyro_z)), float(len(peaks)),
        np.std(acc_mag), np.mean(np.abs(np.diff(gyro_z))),
    ], dtype=np.float32)   # 6 features


# ═════════════════════════════════════════════
# 4. PATIENT-LEVEL FEATURE EXTRACTION
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


def extract_patient_features(windows, binary_labels, metadata,
                              subjects_df, enc_stance, enc_walk):
    X, y, dsets, sids = [], [], [], []
    zero_lat = np.zeros(LATENT_DIM * 3, dtype=np.float32)

    for _, s in subjects_df.iterrows():
        m = (metadata["dataset"] == s["dataset"]) & \
            (metadata["subjectID"] == s["subjectID"])
        if m.sum() == 0:
            continue

        p_win  = windows[m].astype("float32")
        p_meta = metadata[m].reset_index(drop=True)
        p_win -= p_win.mean(axis=(0, 1), keepdims=True)  # per-subject DC removal

        is_stance  = ((p_meta["taskID"] == 0) | (p_meta["taskID"] == 1)).values
        is_walking = ((p_meta["taskID"] == 2) & (p_meta["isTurn"] == 0)).values
        is_turning = ((p_meta["taskID"] == 2) & (p_meta["isTurn"] == 1)).values

        has_s = is_stance.any()
        has_w = is_walking.any()
        has_t = is_turning.any()

        # Latent aggregation: mean + std + max per task encoder
        lat_s = _lat_agg(list(enc_stance.get_latent(p_win[is_stance]).numpy())) if has_s else None
        lat_w = _lat_agg(list(enc_walk.get_latent(p_win[is_walking]).numpy())) if has_w else None
        lat_t = _lat_agg(list(enc_walk.get_latent(p_win[is_turning]).numpy())) if has_t else None

        lat_vec = np.concatenate([
            [float(has_s)], lat_s if lat_s is not None else zero_lat,
            [float(has_w)], lat_w if lat_w is not None else zero_lat,
            [float(has_t)], lat_t if lat_t is not None else zero_lat,
        ])  # 3*(1 + 16*3) = 147

        # Handcrafted aggregation
        hc_s = _agg([stance_features(w)  for w in p_win[is_stance]],  12) if has_s else np.zeros(24, np.float32)
        hc_w = _agg([walking_features(w) for w in p_win[is_walking]], 10) if has_w else np.zeros(20, np.float32)
        hc_t = _agg([turning_features(w) for w in p_win[is_turning]], 6)  if has_t else np.zeros(12, np.float32)

        hc_vec = np.concatenate([[float(has_s), float(has_w), float(has_t)], hc_s, hc_w, hc_t])  # 59

        X.append(np.concatenate([lat_vec, hc_vec]))  # 206 total
        y.append(float(binary_labels[m][0]))
        dsets.append(s["dataset"])
        sids.append(s["subjectID"])

    return np.stack(X), np.array(y), np.array(dsets), np.array(sids)


def split_lat_hc(X):
    lat_block = 3 * (1 + LATENT_DIM * 3)  # 147
    return X[:, :lat_block], X[:, lat_block:]


# ═════════════════════════════════════════════
# 5. CLASSIFIER + EVALUATION
# ═════════════════════════════════════════════

def build_clf():
    """
    Regularized RF to avoid memorization on ~150 patients.
    max_depth=10, min_samples_leaf=3, max_features='sqrt'.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate(name, X_tr, y_tr, X_te, y_te, dsets_te):
    clf = build_clf()
    clf.fit(X_tr, y_tr)

    y_pred     = clf.predict(X_te)
    y_prob     = clf.predict_proba(X_te)[:, 1]
    train_acc  = accuracy_score(y_tr, clf.predict(X_tr))
    acc        = accuracy_score(y_te, y_pred)
    bacc       = balanced_accuracy_score(y_te, y_pred) 
    prec      = precision_score(y_te, y_pred)
    rec       = recall_score(y_te, y_pred)
    f1         = f1_score(y_te, y_pred, average="macro", zero_division=0)
    auc        = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else float("nan")
    cm         = confusion_matrix(y_te, y_pred)

    print(f"\n{'─'*60}")
    print(f"  {name}  ({X_tr.shape[1]} features)")
    print(f"{'─'*60}")
    print(f"  Train acc        : {train_acc:.4f}")
    print(f"  Test acc         : {acc:.4f}")
    print(f"  Balanced acc     : {bacc:.4f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  Macro F1         : {f1:.4f}")
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  Confusion matrix :\n{cm}")
    print(f"\n  Per-dataset accuracy:")
    for ds in np.unique(dsets_te):
        ds_m = dsets_te == ds
        print(f"    {ds:15s}: {accuracy_score(y_te[ds_m], y_pred[ds_m]):.3f}  (n={ds_m.sum()})")

    return {
        "name": name, "clf": clf, "n_feat": X_tr.shape[1],
        "train_acc": train_acc, "acc": acc, "bacc": bacc,
        "prec": prec, "rec": rec,
        "f1": f1, "auc": auc, "cm": cm,
        "y_pred": y_pred, "y_prob": y_prob,
    }


def plot_confusion_matrices(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = r["cm"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r['name']}\nacc={r['acc']:.3f}  AUC={r['auc']:.3f}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Stable", "Unstable"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Stable", "Unstable"], fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrices → {save_path}")


def plot_feature_importance(clf, n_lat, n_hc, save_path, top_k=25):
    imp = clf.feature_importances_
    # Build interpretable names
    lat_names = ([f"Stance_Lat_{i}" for i in range(n_lat // 3)] * 3 +
                 [f"Walk_Lat_{i}"   for i in range(n_lat // 3)] * 3 +
                 [f"Turn_Lat_{i}"   for i in range(n_lat // 3)] * 3)
    hc_names  = (
        ["Stance_has"] + [f"Stance_HC_mean_{i}" for i in range(12)] + [f"Stance_HC_std_{i}" for i in range(12)] +
        ["Walk_has"]   + [f"Walk_HC_mean_{i}"   for i in range(10)] + [f"Walk_HC_std_{i}"   for i in range(10)] +
        ["Turn_has"]   + [f"Turn_HC_mean_{i}"   for i in range(6)]  + [f"Turn_HC_std_{i}"   for i in range(6)]
    )
    # Pad to match actual filtered feature count
    all_names = [f"F{i}" for i in range(len(imp))]

    df = pd.DataFrame({"feature": all_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, top_k * 0.35 + 1))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="#2a6f97")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} feature importances (Combined model)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature importance → {save_path}")

def get_or_train_encoder(task_name, windows, labels_4cls, metadata, s_train, s_val, 
                         task_filter_fn, input_shape, arch_mode):
    """
    Filters data for a specific task and trains either the self-supervised autoencoder 
    or the supervised classifier, saving/loading weights dynamically.
    """
    # 1. Apply patient splits and task filtering
    train_mask = subject_mask(metadata, s_train) & task_filter_fn(metadata)
    val_mask    = subject_mask(metadata, s_val)   & task_filter_fn(metadata)
    
    x_train = windows[train_mask]
    x_val   = windows[val_mask]

    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        raise ValueError(
            f"No windows available for {task_name} encoder: "
            f"train={x_train.shape[0]}, val={x_val.shape[0]}. "
            "Check the patient split and task filter."
        )
    
    weights_filename = f"best_{arch_mode}_{task_name}.weights.h5"
    full_path = os.path.join(CHECKPOINT_PATH, weights_filename)

    # 2. Route to the correct model and training loop based on ARCH_MODE
    if arch_mode == "autoencoder":
        model = ImuEncoder(input_shape=input_shape)
        model(tf.zeros((1,) + input_shape))  # Dummy call to initialize variables
        
        if os.path.exists(full_path):
            print(f"Loading {arch_mode} weights for {task_name}...")
            model.load_weights(full_path)
        else:
            print(f"Training {arch_mode} for {task_name}...")
            train_autoencoder(model, x_train, x_val, CHECKPOINT_PATH, weights_filename)

    elif arch_mode == "classifier":
        y_train = labels_4cls[train_mask]
        y_val   = labels_4cls[val_mask]
        
        # Determine number of classes safely
        n_classes = len(np.unique(labels_4cls)) if len(labels_4cls) > 0 else 4
        
        model = CnnGru(input_shape=input_shape, n_classes=n_classes)
        model(tf.zeros((1,) + input_shape))  # Dummy call to initialize variables
        
        if os.path.exists(full_path):
            print(f"Loading {arch_mode} weights for {task_name}...")
            model.load_weights(full_path)
        else:
            print(f"Training {arch_mode} for {task_name}...")
            train(model, x_train, y_train, x_val, y_val, CHECKPOINT_PATH, weights_filename)
    else:
        raise ValueError(f"Unknown ARCH_MODE: {arch_mode}")

    return model

# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    print("Loading datasets...")
    windows, labels_4cls, binary_labels, metadata = load_data()
    input_shape = windows.shape[1:]

    s_train, s_val, s_test = patient_split(metadata, binary_labels)

    # Train or load the encoder for the "stance" task
    enc_stance = get_or_train_encoder(
        task_name="stance", 
        windows=windows, 
        labels_4cls=labels_4cls, 
        metadata=metadata, 
        s_train=s_train, 
        s_val=s_val,
        task_filter_fn=lambda m: (m["taskID"] < 2),
        input_shape=input_shape,
        arch_mode=ARCH_MODE
    )
    
    # Train or load the encoder for the "walk" task
    enc_walk = get_or_train_encoder(
        task_name="walk", 
        windows=windows, 
        labels_4cls=labels_4cls, 
        metadata=metadata, 
        s_train=s_train, 
        s_val=s_val,
        task_filter_fn=lambda m: (m["taskID"] == 2),
        input_shape=input_shape,
        arch_mode=ARCH_MODE
    )

    print("\nExtracting patient-level features...")
    
    s_fit = pd.concat([s_train, s_val]).reset_index(drop=True)

    X_fit,  y_fit,  dsets_fit,  _         = extract_patient_features(windows, binary_labels, metadata, s_fit,  enc_stance, enc_walk)
    X_test, y_test, dsets_test, sids_test = extract_patient_features(windows, binary_labels, metadata, s_test, enc_stance, enc_walk)

    print(f"Full feature vector : {X_fit.shape[1]} dims")
    print(f"Train patients      : {len(y_fit)}  {dict(zip(*np.unique(y_fit,  return_counts=True)))}")
    print(f"Test patients       : {len(y_test)}  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    X_fit_lat,  X_fit_hc  = split_lat_hc(X_fit)
    X_test_lat, X_test_hc = split_lat_hc(X_test)

    # Save features in csv files for potential future analysis
    pd.DataFrame(X_fit, columns=[f"Feat_{i}" for i in range(X_fit.shape[1])]).to_csv(os.path.join(CHECKPOINT_PATH, f"train_features_{ARCH_MODE}.csv"), index=False)
    pd.DataFrame(X_test, columns=[f"Feat_{i}" for i in range(X_test.shape[1])]).to_csv(os.path.join(CHECKPOINT_PATH, f"test_features_{ARCH_MODE}.csv"), index=False)   
    
    def apply_variance_threshold(Xtr, Xte):
        v = VarianceThreshold(threshold=1e-6)
        Xtr_f = v.fit_transform(Xtr)
        Xte_f = v.transform(Xte)
        return Xtr_f, Xte_f, v

    X_fit_lat_f,  X_test_lat_f,  v_lat  = apply_variance_threshold(X_fit_lat,  X_test_lat)
    X_fit_hc_f,   X_test_hc_f,   v_hc   = apply_variance_threshold(X_fit_hc,   X_test_hc)
    X_fit_full_f, X_test_full_f, v_full = apply_variance_threshold(X_fit,      X_test)

    print("\n" + "═" * 60)
    print("  ABLATION — Binary classification (stable vs unstable)")
    print("═" * 60)

    res_hc   = evaluate("Handcrafted only",    X_fit_hc_f,   y_fit, X_test_hc_f,   y_test, dsets_test)
    res_lat  = evaluate("Latent only",         X_fit_lat_f,  y_fit, X_test_lat_f,  y_test, dsets_test)
    res_full = evaluate("Combined (lat + HC)", X_fit_full_f, y_fit, X_test_full_f, y_test, dsets_test)

    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    
    summary = pd.DataFrame([{
        "Feature set"  : r["name"],
        "N features"   : r["n_feat"],
        "Train acc"    : round(r["train_acc"], 4),
        "Test acc"     : round(r["acc"],       4),
        "Balanced acc" : round(r["bacc"],      4),
        "Precision"    : round(r["prec"],      4),
        "Recall"       : round(r["rec"],       4),
        "Macro F1"     : round(r["f1"],        4),
        "ROC-AUC"      : round(r["auc"],       4),
    } for r in [res_hc, res_lat, res_full]])
    
    print(summary.to_string(index=False))

    # Feature importance plot
    plot_feature_importance(
        res_full["clf"],
        n_lat=X_fit_lat_f.shape[1],
        n_hc=X_fit_hc_f.shape[1],
        save_path=os.path.join(RESULTS_PATH, f"feature_importance_{ARCH_MODE}.png"),
    )

    # Save outputs
    summary.to_csv(os.path.join(RESULTS_PATH, f"ablation_results_{ARCH_MODE}.csv"), index=False)

    pred_df = pd.DataFrame({
        "subjectID"         : sids_test,
        "dataset"           : dsets_test,
        "y_true"            : y_test.astype(int),
        "y_pred_hc"         : res_hc["y_pred"].astype(int),
        "y_pred_lat"        : res_lat["y_pred"].astype(int),
        "y_pred_full"       : res_full["y_pred"].astype(int),
        "prob_unstable_hc"  : res_hc["y_prob"],
        "prob_unstable_full": res_full["y_prob"],
    })
    pred_df.to_excel(os.path.join(RESULTS_PATH, f"predictions_{ARCH_MODE}.xlsx"), index=False)

    plot_confusion_matrices(
        [res_hc, res_lat, res_full],
        os.path.join(RESULTS_PATH, f"confusion_matrices_{ARCH_MODE}.png"),
    )


if __name__ == "__main__":
    main()