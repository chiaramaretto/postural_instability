import tensorflow as tf
from model import CnnGru, ImuEncoder
import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split


from train import train_autoencoder, train_classifier

# ── Config ────────────────────────────────────────────────────────────────────
from params import TARGET_HZ, LATENT_DIM, TARGET_DOMAIN, RANDOM_STATE

DATA_PATH       = "data/preprocessed_data/"
CHECKPOINT_PATH = "models/"
RESULTS_PATH    = "results/"
CORAL_REG       = 1e-2              # regularisation for CORAL covariance
MIN_CORAL_SAMPLES = 8               # minimum samples per domain for CORAL

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ══════════════════════════════════════════════════════════════════════════════
# UMAP
# ══════════════════════════════════════════════════════════════════════════════

def _lazy_umap():
    try:
        import umap
        return umap
    except ImportError:
        try:
            import umap.umap_ as umap
            return umap
        except ImportError:
            raise ImportError("Install umap-learn to generate UMAP plots.")


def make_umap_plot(df, output_dir, mode_tag, seed=RANDOM_STATE):
    try:
        umap = _lazy_umap()
    except ImportError as e:
        print(f"  [UMAP skipped] {e}")
        return None

    feat_cols  = [c for c in df.columns if c.startswith("Feat_")]
    lat_cols   = feat_cols[: 4 * LATENT_DIM]
    valid_mask = df[lat_cols].notna().all(axis=1)
    df_v       = df[valid_mask]

    if len(df_v) < 5:
        print("  [UMAP skipped] too few valid rows.")
        return None

    embedding = umap.UMAP(n_components=2, random_state=seed).fit_transform(
        df_v[lat_cols].to_numpy(dtype=np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ds in pd.unique(df_v["dataset"]):
        m = df_v["dataset"] == ds
        axes[0].scatter(embedding[m, 0], embedding[m, 1], label=ds, alpha=0.65, s=35, fontsize=15)
    axes[0].set_title("Latent space by dataset", fontsize=18)
    axes[0].legend(fontsize=13, loc="upper left", bbox_to_anchor=(0, -0.12),
               ncol=2, borderaxespad=0)


    for lbl, col, name in [(0, "steelblue", "HC"), (1, "tomato", "PD")]:
        m = df_v["y_true"] == lbl
        axes[1].scatter(embedding[m, 0], embedding[m, 1], c=col, label=name, alpha=0.65, s=35, fontsize=15)
    axes[1].set_title("Latent space by class", fontsize=18)
    axes[1].legend(fontsize=13, loc="upper left", bbox_to_anchor=(0, -0.12),
               ncol=2, borderaxespad=0)

    fig.suptitle(f"UMAP — {mode_tag}", y=1.02, fontsize=18)
    fig.tight_layout(rect=[0, 0.08, 1, 1])  # lascia spazio in basso per le legende
    out = os.path.join(output_dir, f"umap_latent_{mode_tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  UMAP saved -> {out}")
    return out


_THESIS_STYLE = {
    "font.family":           "sans-serif",
    "font.size":             18,
    "axes.titlesize":        18,
    "axes.titleweight":      "bold",
    "axes.labelsize":        18,
    "xtick.labelsize":       18,
    "ytick.labelsize":       18,
    "legend.fontsize":       18,
    "legend.title_fontsize": 18,
    "legend.framealpha":     0.85,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "figure.dpi":            120,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
}

# Wong colorblind-safe palette (7 colours)
_WONG = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#F0E442"]


def make_umap_combined(dfs_by_variant, output_dir, arch_mode, seed=RANDOM_STATE):
    """
    Single 1×3 figure: UMAP latent space coloured by dataset for
    baseline / CORAL / MMD side by side.

    dfs_by_variant: dict {"baseline": df, "coral": df, "mmd": df}
                    each df is train+test concatenated.
    """
    try:
        umap_lib = _lazy_umap()
    except ImportError as e:
        print(f"  [UMAP combined] {e}")
        return None

    plt.rcParams.update(_THESIS_STYLE)

    variant_titles = {"baseline": "Baseline", "coral": "CORAL", "mmd": "MMD"}

    # Consistent dataset colours across all panels
    all_datasets = sorted({
        ds
        for df in dfs_by_variant.values() if df is not None
        for ds in pd.unique(df["dataset"])
    })
    ds_colors = {ds: _WONG[i % len(_WONG)] for i, ds in enumerate(all_datasets)}

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for ax, variant in zip(axes, ["baseline", "coral", "mmd"]):
        df = dfs_by_variant.get(variant)
        ax.set_title(variant_titles[variant])
        ax.grid(True, alpha=0.35, linestyle="--", zorder=0)
        ax.set_xlabel("UMAP 1", fontsize=16)
        ax.set_ylabel("UMAP 2", fontsize=16)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if df is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        feat_cols = [c for c in df.columns if c.startswith("Feat_")]
        lat_cols  = feat_cols[: 4 * LATENT_DIM]
        df_v = df[df[lat_cols].notna().all(axis=1)]

        if len(df_v) < 5:
            ax.text(0.5, 0.5, "Too few samples", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        embedding = umap_lib.UMAP(n_components=2, random_state=seed).fit_transform(
            df_v[lat_cols].to_numpy(dtype=np.float32)
        )

        for ds in all_datasets:
            m = df_v["dataset"].values == ds
            if m.sum() == 0:
                continue
            ax.scatter(embedding[m, 0], embedding[m, 1],
                       c=ds_colors[ds], label=ds,
                       alpha=0.75, s=50, linewidths=0, zorder=3)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=ds_colors[ds], markersize=12, label=ds)
        for ds in all_datasets
    ]
    fig.legend(handles=legend_handles, title="Dataset",
               loc="lower center", ncol=len(all_datasets),
               bbox_to_anchor=(0.5, -0.15), framealpha=0.85, fontsize=16, title_fontsize=16)

    fig.suptitle(f"UMAP – Latent Space by Dataset  [{arch_mode}]",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = os.path.join(output_dir, f"umap_combined_{arch_mode}_seed{seed}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined UMAP saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Data loading & splitting
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    windows      = np.load(os.path.join(DATA_PATH, "windows.npy"))
    labels_raw   = np.load(os.path.join(DATA_PATH, "labels.npy")).astype(np.float32)
    metadata     = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
    labels_4cls  = np.clip(labels_raw, 0, 3).astype(np.int32)
    binary_labels = (labels_4cls >= 1).astype(np.float32)
    print(f"Windows      : {windows.shape}")
    print(f"4-class dist : {dict(zip(*np.unique(labels_4cls, return_counts=True)))}")
    print(f"Binary dist  : {dict(zip(*np.unique(binary_labels, return_counts=True)))}")
    return windows, labels_4cls, binary_labels, metadata


def subject_mask(metadata, subjects_df):
    meta_idx = pd.MultiIndex.from_frame(metadata[["dataset", "subjectID"]])
    subj_idx = pd.MultiIndex.from_frame(subjects_df[["dataset", "subjectID"]])
    return meta_idx.isin(subj_idx)


def patient_split(metadata, binary_labels, seed=RANDOM_STATE):
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
        s_trval, s_test = train_test_split(cls, test_size=0.25, random_state=seed)
        s_tr, s_val     = train_test_split(s_trval, test_size=0.2, random_state=seed)
        train_list.append(s_tr)
        val_list.append(s_val)
        test_list.append(s_test)

    s_train = pd.concat(train_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    s_val   = pd.concat(val_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    s_test  = pd.concat(test_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    for df in (s_train, s_val, s_test):
        df.drop(columns=["bin_label"], inplace=True)
    print(f"\nSplit — train: {len(s_train)}, val: {len(s_val)}, test: {len(s_test)} patients")
    return s_train, s_val, s_test


def ensure_validation_domain_coverage(s_train, s_val, seed=RANDOM_STATE):
    missing = [ds for ds in pd.unique(s_train["dataset"])
               if ds not in set(pd.unique(s_val["dataset"]))]
    if not missing:
        return s_train, s_val
    s_train, s_val = s_train.copy(), s_val.copy()
    moved = []
    for ds in missing:
        cands = s_train[s_train["dataset"] == ds]
        if len(cands) <= 1:
            continue
        row = cands.iloc[[0]]
        s_train = s_train.drop(index=row.index)
        moved.append(row)
    if moved:
        s_val   = pd.concat([s_val] + moved, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        s_train = s_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    return s_train, s_val


# ══════════════════════════════════════════════════════════════════════════════
# Handcrafted features
# ══════════════════════════════════════════════════════════════════════════════

def _bandpass(signal, lo=0.03, hi=1.0, fs=TARGET_HZ, order=2):
    nyq = 0.5 * fs
    lo_n = np.clip(lo / nyq, 1e-5, 0.99)
    hi_n = np.clip(hi / nyq, lo_n + 1e-5, 0.99)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    if len(signal) < max(len(b), len(a)) * 3:
        return signal
    return filtfilt(b, a, signal)


def stance_features(window, fs=TARGET_HZ):
    acc_ml = _bandpass(window[:, 1], fs=fs)
    acc_ap = _bandpass(window[:, 2], fs=fs)
    cov    = np.cov(acc_ap, acc_ml)
    det    = np.linalg.det(cov)
    sway   = np.pi * 5.991 * np.sqrt(max(det, 0.0))
    lat_dom = np.var(acc_ml) / (np.var(acc_ap) + 1e-8)
    fft_ml  = np.abs(np.fft.rfft(acc_ml)) ** 2
    freqs   = np.fft.rfftfreq(len(acc_ml), d=1.0 / fs)
    p_tot   = np.sum(fft_ml) + 1e-10
    p_sway  = np.sum(fft_ml[(freqs >= 0.1) & (freqs <= 0.5)]) / p_tot
    p_trem  = np.sum(fft_ml[(freqs >= 8)   & (freqs <= 12)]) / p_tot
    return np.array([sway, lat_dom, p_sway, p_trem], dtype=np.float32)


def walking_features(window, fs=TARGET_HZ):
    acc_v   = window[:, 0]
    acc_ap  = window[:, 2]
    jerk    = np.linalg.norm(np.diff(window[:, :3], axis=0) * fs, axis=1)
    n_jerk  = np.sum(jerk) / (len(acc_v) / fs + 1e-8)
    peaks, _ = find_peaks(acc_v, distance=int(fs * 0.3), prominence=np.std(acc_v) * 0.3)
    step_cv = 0.0
    if len(peaks) > 1:
        ivs = np.diff(peaks) / fs
        step_cv = np.std(ivs) / (np.mean(ivs) + 1e-8)
    fft_ap  = np.abs(np.fft.rfft(acc_ap))
    freqs   = np.fft.rfftfreq(len(acc_ap), d=1.0 / fs)
    valid   = (freqs > 0.5) & (freqs < 4.0)
    dom_f   = float(freqs[np.argmax(fft_ap[valid])]) if valid.any() else 0.0
    return np.array([n_jerk, step_cv, dom_f if np.isfinite(dom_f) else 0.0], dtype=np.float32)


def _agg(rows, n_feat):
    if len(rows) == 0:
        return np.zeros(n_feat * 2, dtype=np.float32)
    arr = np.array(rows, dtype=np.float32)
    return np.concatenate([arr.mean(0), arr.std(0)])


def _lat_agg(rows):
    if len(rows) == 0:
        return None
    arr = np.array(rows, dtype=np.float32)
    n   = arr.shape[0]
    if n >= 3:
        t   = np.arange(n, dtype=np.float32)
        tc  = t - t.mean()
        tv  = np.dot(tc, tc) + 1e-8
        slp = np.dot(tc, arr) / tv
    else:
        slp = np.zeros(arr.shape[1], dtype=np.float32)
    return np.concatenate([arr.mean(0), arr.std(0), arr.max(0), slp])


# ══════════════════════════════════════════════════════════════════════════════
# Patient-level feature extraction (46 dims)
# ══════════════════════════════════════════════════════════════════════════════

def extract_patient_features(windows, binary_labels, labels_4cls,
                              metadata, subjects_df, enc_shared):
    X, y, y4, dsets, sids = [], [], [], [], []
    nan_lat  = np.full(LATENT_DIM * 4, np.nan, dtype=np.float32)
    nan_hc_s = np.full(8, np.nan, dtype=np.float32)
    nan_hc_w = np.full(6, np.nan, dtype=np.float32)

    for _, s in subjects_df.iterrows():
        m = (metadata["dataset"] == s["dataset"]) & (metadata["subjectID"] == s["subjectID"])
        if m.sum() == 0:
            continue
        p_win  = windows[m].astype("float32")
        p_meta = metadata[m].reset_index(drop=True)
        p_win -= p_win.mean(axis=(0, 1), keepdims=True)
        p_win /= (p_win.std(axis=(0, 1), keepdims=True) + 1e-8)

        is_stance  = ((p_meta["taskID"] == 0) | (p_meta["taskID"] == 1)).values
        is_walking = (p_meta["taskID"] == 2).values
        is_all     = is_stance | is_walking

        lat  = _lat_agg(list(enc_shared.get_latent(p_win[is_all]).numpy()))  if is_all.any()    else nan_lat
        hc_s = _agg([stance_features(w)  for w in p_win[is_stance]],  4)    if is_stance.any() else nan_hc_s
        hc_w = _agg([walking_features(w) for w in p_win[is_walking]], 3)    if is_walking.any() else nan_hc_w

        X.append(np.concatenate([lat, hc_s, hc_w]))
        y.append(float(binary_labels[m][0]))
        y4.append(int(labels_4cls[m][0]))
        dsets.append(s["dataset"])
        sids.append(s["subjectID"])

    return np.stack(X), np.array(y), np.array(y4), np.array(dsets), np.array(sids)


# ══════════════════════════════════════════════════════════════════════════════
# Encoder training / loading
# ══════════════════════════════════════════════════════════════════════════════

def get_or_train_encoder(windows, labels_4cls, metadata,
                         s_train, s_val, input_shape, arch_mode, seed=RANDOM_STATE):
    train_mask = subject_mask(metadata, s_train) & (metadata["taskID"] <= 2)
    val_mask   = subject_mask(metadata, s_val)   & (metadata["taskID"] <= 2)
    x_train, x_val = windows[train_mask], windows[val_mask]

    weights_fn = f"best_{arch_mode}_shared_seed{seed}.weights.h5"
    full_path  = os.path.join(CHECKPOINT_PATH, weights_fn)

    if arch_mode == "autoencoder":
        model = ImuEncoder(input_shape=input_shape, latent_dim=LATENT_DIM)
        model(tf.zeros((1,) + input_shape))
        if os.path.exists(full_path):
            print(f"  Loading autoencoder weights...")
            model.load_weights(full_path)
        else:
            print(f"  Training autoencoder...")
            train_autoencoder(model, x_train, x_val, CHECKPOINT_PATH, weights_fn)

    elif arch_mode == "classifier":
        y_tr = labels_4cls[train_mask]
        y_va = labels_4cls[val_mask]
        model = CnnGru(input_shape=input_shape,
                       n_classes=len(np.unique(labels_4cls)))
        model(tf.zeros((1,) + input_shape))
        if os.path.exists(full_path):
            print(f"  Loading classifier weights...")
            model.load_weights(full_path)
        else:
            print(f"  Training classifier...")
            train_classifier(model, x_train, y_tr, x_val, y_va,
                             CHECKPOINT_PATH, weights_fn)
    else:
        raise ValueError(f"Unknown arch_mode: {arch_mode}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Domain adaptation — CORAL
# ══════════════════════════════════════════════════════════════════════════════

def fit_coral(X_fit, domains_fit, target_domain=TARGET_DOMAIN,
              reg=CORAL_REG, min_samples=MIN_CORAL_SAMPLES):
    valid = ~np.isnan(X_fit).any(axis=1)
    n_feat = X_fit.shape[1]

    # ── Target covariance (wearpd) ────────────────────────────────────────────
    tgt_mask = valid & (domains_fit == target_domain)
    if tgt_mask.sum() < min_samples:
        print(f"  [CORAL] Target '{target_domain}' has only {tgt_mask.sum()} samples "
              f"(min={min_samples}) — CORAL disabled.")
        return {}

    X_tgt  = X_fit[tgt_mask]
    mu_t   = X_tgt.mean(axis=0)
    cov_t  = np.cov(X_tgt - mu_t, rowvar=False) + reg * np.eye(n_feat)
    try:
        cov_t_half = np.real(fractional_matrix_power(cov_t, 0.5))
    except Exception as e:
        print(f"  [CORAL] Failed to compute target cov^0.5: {e}. CORAL disabled.")
        return {}

    stats = {
        "target_domain": target_domain,
        "mu_target":     mu_t,
        "cov_t_half":    cov_t_half,
        "sources":       {},
    }

    # ── Source covariances ────────────────────────────────────────────────────
    for ds in np.unique(domains_fit):
        if ds == target_domain:
            continue
        src_mask = valid & (domains_fit == ds)
        n = src_mask.sum()
        if n < min_samples:
            print(f"  [CORAL] Source '{ds}' has only {n} samples "
                  f"(min={min_samples}) — skipped.")
            continue
        X_src = X_fit[src_mask]
        mu_s  = X_src.mean(axis=0)
        # Check for NaN/Inf before computing matrix power
        cov_s = np.cov(X_src - mu_s, rowvar=False) + reg * np.eye(n_feat)
        if not np.isfinite(cov_s).all():
            print(f"  [CORAL] Source '{ds}' covariance has NaN/Inf — skipped.")
            continue
        try:
            cov_s_inv_half = np.real(fractional_matrix_power(cov_s, -0.5))
        except Exception as e:
            print(f"  [CORAL] Source '{ds}' cov^-0.5 failed: {e} — skipped.")
            continue
        stats["sources"][ds] = {"mu_s": mu_s, "cov_s_inv_half": cov_s_inv_half}
        print(f"  [CORAL] Source '{ds}': {n} samples — fitted OK.")

    return stats


def transform_coral(X, domains, stats):

    if not stats:
        return np.copy(X)

    X_out     = np.copy(X)
    mu_t      = stats["mu_target"]
    cov_t_half = stats["cov_t_half"]
    tgt       = stats["target_domain"]

    for ds in np.unique(domains):
        if ds == tgt:
            continue
        if ds not in stats["sources"]:
            continue
        idx = np.where((domains == ds) & ~np.isnan(X[:, 0]))[0]
        if len(idx) == 0:
            continue
        mu_s          = stats["sources"][ds]["mu_s"]
        cov_s_inv_half = stats["sources"][ds]["cov_s_inv_half"]
        X_ds          = X[idx]
        X_out[idx]    = (X_ds - mu_s) @ cov_s_inv_half @ cov_t_half + mu_t

    return X_out


# ══════════════════════════════════════════════════════════════════════════════
# Domain adaptation — MMD mean shift
# ══════════════════════════════════════════════════════════════════════════════

def fit_mmd_shift(X_fit, domains_fit, target_domain=TARGET_DOMAIN,
                  min_samples=5):
    """
    Fit MMD mean-shift on training set.
    For each source domain, the shift vector = mu_target - mu_source.
    Very stable even with few samples (requires only mean estimation).
    """
    valid = ~np.isnan(X_fit).any(axis=1)
    tgt_mask = valid & (domains_fit == target_domain)

    if tgt_mask.sum() < min_samples:
        print(f"  [MMD] Target '{target_domain}' has only {tgt_mask.sum()} samples — MMD disabled.")
        return {}

    mu_t = X_fit[tgt_mask].mean(axis=0)
    stats = {"target_domain": target_domain, "mu_target": mu_t, "shifts": {}}

    for ds in np.unique(domains_fit):
        if ds == target_domain:
            continue
        src_mask = valid & (domains_fit == ds)
        n = src_mask.sum()
        if n < min_samples:
            print(f"  [MMD] Source '{ds}' has only {n} samples — skipped.")
            continue
        mu_s = X_fit[src_mask].mean(axis=0)
        shift = mu_t - mu_s
        stats["shifts"][ds] = shift
        print(f"  [MMD] Source '{ds}': {n} samples, "
              f"shift norm={np.linalg.norm(shift):.4f}")

    return stats


def transform_mmd_shift(X, domains, stats):

    if not stats:
        return np.copy(X)

    X_out = np.copy(X)
    for ds, shift in stats["shifts"].items():
        idx = np.where((domains == ds) & ~np.isnan(X[:, 0]))[0]
        if len(idx) == 0:
            continue
        X_out[idx] += shift

    return X_out


# ══════════════════════════════════════════════════════════════════════════════
# CSV builder
# ══════════════════════════════════════════════════════════════════════════════

def build_df(X, y, y4, dsets, sids):
    df = pd.DataFrame(X, columns=[f"Feat_{i}" for i in range(X.shape[1])])
    df["y_true"]      = y
    df["y_true_4cls"] = y4
    df["dataset"]     = dsets
    df["subjectID"]   = sids
    return df


def save_variant(train_df, test_df, arch_mode, da_tag, seed=RANDOM_STATE):
    tag = f"{arch_mode}_{da_tag}_seed{seed}" if da_tag else f"{arch_mode}_seed{seed}"
    tr_path = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{tag}.csv")
    te_path = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{tag}.csv")
    train_df.to_csv(tr_path, index=False)
    test_df.to_csv(te_path,  index=False)
    print(f"  Saved train -> {tr_path}")
    print(f"  Saved test  -> {te_path}")
    return tag


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(arch_mode="autoencoder", seed=RANDOM_STATE):
    set_seeds(seed)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH,    exist_ok=True)

    print(f"Seed: {seed}")
    print("Loading datasets...")
    windows, labels_4cls, binary_labels, metadata = load_data()
    input_shape = windows.shape[1:]

    s_train, s_val, s_test = patient_split(metadata, binary_labels, seed=seed)
    s_train, s_val = ensure_validation_domain_coverage(s_train, s_val, seed=seed)

    # ── Train / load shared encoder ───────────────────────────────────────────
    enc = get_or_train_encoder(windows, labels_4cls, metadata,
                               s_train, s_val, input_shape, arch_mode, seed=seed)

    # ── Extract raw features ──────────────────────────────────────────────────
    s_fit = pd.concat([s_train, s_val]).reset_index(drop=True)

    print("\nExtracting patient-level features (fit set)...")
    X_fit,  y_fit,  y4_fit,  d_fit,  s_fit_ids = extract_patient_features(
        windows, binary_labels, labels_4cls, metadata, s_fit,  enc)

    print("Extracting patient-level features (test set)...")
    X_test, y_test, y4_test, d_test, s_test_ids = extract_patient_features(
        windows, binary_labels, labels_4cls, metadata, s_test, enc)

    n_valid_fit  = (~np.isnan(X_fit).any(axis=1)).sum()
    n_valid_test = (~np.isnan(X_test).any(axis=1)).sum()
    print(f"\nFeature dims : {X_fit.shape[1]}")
    print(f"Fit  patients: {len(X_fit)}  ({n_valid_fit} fully valid)")
    print(f"Test patients: {len(X_test)} ({n_valid_test} fully valid)")

    tr_df_raw = build_df(X_fit,  y_fit,  y4_fit,  d_fit,  s_fit_ids)
    te_df_raw = build_df(X_test, y_test, y4_test, d_test, s_test_ids)

    # ── Variant 1: baseline (no DA) ───────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f" Variant: BASELINE (no domain adaptation)")
    print(f"{'─'*55}")
    tag_base = save_variant(tr_df_raw, te_df_raw, arch_mode, "baseline", seed=seed)
    try:
        make_umap_plot(pd.concat([tr_df_raw, te_df_raw], ignore_index=True),
                       RESULTS_PATH, tag_base, seed=seed)
    except Exception as e:
        print(f"  [UMAP] {e}")

    # ── Variant 2: CORAL ──────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f" Variant: CORAL (target={TARGET_DOMAIN})")
    print(f"{'─'*55}")
    coral_stats = fit_coral(X_fit, d_fit)

    X_fit_coral  = transform_coral(X_fit,  d_fit,  coral_stats)
    X_test_coral = transform_coral(X_test, d_test, coral_stats)

    tr_df_coral = build_df(X_fit_coral,  y_fit,  y4_fit,  d_fit,  s_fit_ids)
    te_df_coral = build_df(X_test_coral, y_test, y4_test, d_test, s_test_ids)
    tag_coral   = save_variant(tr_df_coral, te_df_coral, arch_mode, "coral", seed=seed)
    try:
        make_umap_plot(pd.concat([tr_df_coral, te_df_coral], ignore_index=True),
                       RESULTS_PATH, tag_coral, seed=seed)
    except Exception as e:
        print(f"  [UMAP] {e}")

    # ── Variant 3: MMD mean shift ─────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f" Variant: MMD mean-shift (target={TARGET_DOMAIN})")
    print(f"{'─'*55}")
    mmd_stats = fit_mmd_shift(X_fit, d_fit)

    X_fit_mmd  = transform_mmd_shift(X_fit,  d_fit,  mmd_stats)
    X_test_mmd = transform_mmd_shift(X_test, d_test, mmd_stats)

    tr_df_mmd = build_df(X_fit_mmd,  y_fit,  y4_fit,  d_fit,  s_fit_ids)
    te_df_mmd = build_df(X_test_mmd, y_test, y4_test, d_test, s_test_ids)
    tag_mmd   = save_variant(tr_df_mmd, te_df_mmd, arch_mode, "mmd", seed=seed)
    try:
        make_umap_plot(pd.concat([tr_df_mmd, te_df_mmd], ignore_index=True),
                       RESULTS_PATH, tag_mmd, seed=seed)
    except Exception as e:
        print(f"  [UMAP] {e}")

    # ── Combined UMAP (all three variants, one figure) ───────────────────────
    try:
        make_umap_combined(
            {
                "baseline": pd.concat([tr_df_raw,   te_df_raw],   ignore_index=True),
                "coral":    pd.concat([tr_df_coral,  te_df_coral], ignore_index=True),
                "mmd":      pd.concat([tr_df_mmd,    te_df_mmd],   ignore_index=True),
            },
            RESULTS_PATH, arch_mode, seed=seed,
        )
    except Exception as e:
        print(f"  [UMAP combined] {e}")

    print(f"\n{'═'*55}")
    print(f" Feature extraction complete for arch_mode={arch_mode}, seed={seed}")
    print(f" Saved variants: baseline, coral, mmd")
    print(f"{'═'*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch-mode",
                        choices=["autoencoder", "classifier"],
                        default="autoencoder")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    main(arch_mode=args.arch_mode, seed=args.seed)
