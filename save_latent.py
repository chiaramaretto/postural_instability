"""
save_latents_per_window.py
──────────────────────────
Estrae le slope OLS per ogni soggetto/task/dimensione latente
e produce UN SOLO report aggregato per capire se il trend
è informativo (differenza stabile vs instabile).

Output:
  models/latent_per_window_<arch>.csv        (slope per soggetto)
  results/trend_analysis_<arch>.png          (1 figura, 2 task)
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, mannwhitneyu

from model import CnnGru, ImuEncoder

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH       = "posturalInstability/data/preprocessed_data/"
CHECKPOINT_PATH = "posturalInstability/models/"
RESULTS_PATH    = "posturalInstability/results/"
LATENT_DIM      = 8
ARCH_MODE       = "classifier_mmd"  # "autoencoder" or "classifier"
INPUT_SHAPE     = (320, 6)

os.makedirs(RESULTS_PATH, exist_ok=True)

LABEL_COLOR = {0: "#2196F3", 1: "#F44336"}
LABEL_NAME  = {0: "Stable", 1: "Unstable"}

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_model(arch_mode, task_name):
    # Support weight filenames with the `_mmd` token placed after the task
    # e.g. best_autoencoder_stance_mmd.weights.h5
    if arch_mode.endswith("_mmd"):
        base = arch_mode.replace("_mmd", "")
        weights = os.path.join(CHECKPOINT_PATH, f"best_{base}_{task_name}_mmd.weights.h5")
    else:
        weights = os.path.join(CHECKPOINT_PATH, f"best_{arch_mode}_{task_name}.weights.h5")

    # Choose model class based on base architecture (handles _mmd suffix)
    if arch_mode.startswith("autoencoder"):
        model = ImuEncoder(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM)
    else:
        model = CnnGru(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM)

    model(tf.zeros((1,) + INPUT_SHAPE))
    model.load_weights(weights)
    print(f"  Loaded {arch_mode}/{task_name} -> {os.path.basename(weights)}")
    return model

def compute_slope(values):
    n = len(values)
    if n < 3:
        return 0.0, 0.0, 1.0
    t = np.arange(n, dtype=float)
    slope, _, r, p, _ = linregress(t, values)
    return float(slope), float(r**2), float(p)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    windows      = np.load(os.path.join(DATA_PATH, "windows.npy"))
    metadata     = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
    raw_labels   = np.load(os.path.join(DATA_PATH, "labels.npy")).astype(np.int32)
    binary_labels = (raw_labels >= 1).astype(np.int32)

    enc_stance = load_model(ARCH_MODE, "stance")
    enc_walk   = load_model(ARCH_MODE, "walk")

    subjects = metadata[["dataset", "subjectID"]].drop_duplicates()
    records  = []

    print(f"\nComputing slopes for {len(subjects)} subjects...")
    for _, s in subjects.iterrows():
        sid, ds = s["subjectID"], s["dataset"]
        m = (metadata["dataset"] == ds) & (metadata["subjectID"] == sid)

        p_win  = windows[m].astype("float32")
        p_meta = metadata[m].reset_index(drop=True)
        p_win -= p_win.mean(axis=(0, 1), keepdims=True)

        sort_idx = p_meta["window_id"].argsort().values
        p_win    = p_win[sort_idx]
        p_meta   = p_meta.iloc[sort_idx].reset_index(drop=True)
        label    = int(binary_labels[m][0])

        for task_name, enc, mask_fn in [
            ("stance", enc_stance, lambda mm: (mm["taskID"] < 2).values),
            ("walk",   enc_walk,   lambda mm: ((mm["taskID"] == 2) & (mm["isTurn"] == 0)).values),
        ]:
            mask = mask_fn(p_meta)
            if not mask.any():
                continue

            lats = enc.get_latent(p_win[mask]).numpy()
            rec  = {"subjectID": sid, "dataset": ds, "task": task_name,
                    "label": label, "n_windows": len(lats)}
            for d in range(LATENT_DIM):
                slope, r2, p = compute_slope(lats[:, d])
                rec[f"slope_{d}"] = slope
                rec[f"r2_{d}"]    = r2
                rec[f"p_{d}"]     = p
            records.append(rec)

    df = pd.DataFrame(records)
    csv_path = os.path.join(CHECKPOINT_PATH, f"latent_slopes_{ARCH_MODE}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Slopes CSV saved -> {csv_path}")

    # ── ANALISI AGGREGATA ────────────────────────────────────────────────────
    tasks = df["task"].unique()
    slope_cols = [f"slope_{d}" for d in range(LATENT_DIM)]

    fig, axes = plt.subplots(len(tasks), 2, figsize=(14, 5 * len(tasks)))
    if len(tasks) == 1:
        axes = axes[None, :]

    print("\n── Mann-Whitney U test: slope stable vs unstable ──")
    for t_idx, task in enumerate(tasks):
        sub = df[df["task"] == task]
        stable   = sub[sub["label"] == 0]
        unstable = sub[sub["label"] == 1]

        # ── Boxplot slopes per dimensione ────────────────────────────────
        ax = axes[t_idx, 0]
        positions_stable   = np.arange(LATENT_DIM) - 0.2
        positions_unstable = np.arange(LATENT_DIM) + 0.2

        bp1 = ax.boxplot(
            [stable[c].dropna().values for c in slope_cols],
            positions=positions_stable, widths=0.35,
            patch_artist=True, medianprops=dict(color="white", linewidth=2),
            boxprops=dict(facecolor=LABEL_COLOR[0], alpha=0.7),
            whiskerprops=dict(color=LABEL_COLOR[0]),
            capprops=dict(color=LABEL_COLOR[0]),
            flierprops=dict(marker="o", markersize=3, color=LABEL_COLOR[0], alpha=0.4)
        )
        bp2 = ax.boxplot(
            [unstable[c].dropna().values for c in slope_cols],
            positions=positions_unstable, widths=0.35,
            patch_artist=True, medianprops=dict(color="white", linewidth=2),
            boxprops=dict(facecolor=LABEL_COLOR[1], alpha=0.7),
            whiskerprops=dict(color=LABEL_COLOR[1]),
            capprops=dict(color=LABEL_COLOR[1]),
            flierprops=dict(marker="o", markersize=3, color=LABEL_COLOR[1], alpha=0.4)
        )

        # Asterischi per significatività
        print(f"\n  Task: {task}")
        for d, col in enumerate(slope_cols):
            s_vals = stable[col].dropna().values
            u_vals = unstable[col].dropna().values
            if len(s_vals) > 1 and len(u_vals) > 1:
                stat, p = mannwhitneyu(s_vals, u_vals, alternative="two-sided")
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"    lat_{d}: p={p:.4f}  {sig}")
                if p < 0.05:
                    y_max = max(
                        np.percentile(s_vals, 95) if len(s_vals) else 0,
                        np.percentile(u_vals, 95) if len(u_vals) else 0
                    )
                    ax.text(d, y_max * 1.05, sig, ha="center", fontsize=9,
                            color="black", fontweight="bold")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(LATENT_DIM))
        ax.set_xticklabels([f"lat_{d}" for d in range(LATENT_DIM)], fontsize=8)
        ax.set_title(f"{task.upper()} — OLS slope distribution", fontsize=10)
        ax.set_ylabel("Slope (per window)")
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
                  [LABEL_NAME[0], LABEL_NAME[1]], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # ── Effect size (Cohen's d) per dimensione ───────────────────────
        ax2 = axes[t_idx, 1]
        cohens_d = []
        for col in slope_cols:
            s_vals = stable[col].dropna().values
            u_vals = unstable[col].dropna().values
            if len(s_vals) > 1 and len(u_vals) > 1:
                pooled_std = np.sqrt(
                    (np.var(s_vals, ddof=1) * (len(s_vals) - 1) +
                     np.var(u_vals, ddof=1) * (len(u_vals) - 1)) /
                    (len(s_vals) + len(u_vals) - 2)
                ) + 1e-8
                d = (np.mean(u_vals) - np.mean(s_vals)) / pooled_std
            else:
                d = 0.0
            cohens_d.append(d)

        colors = ["#E53935" if abs(d) > 0.5 else "#FB8C00" if abs(d) > 0.3 else "#9E9E9E"
                  for d in cohens_d]
        bars = ax2.bar(range(LATENT_DIM), cohens_d, color=colors, alpha=0.8, edgecolor="white")
        ax2.axhline(0,    color="gray",   linewidth=0.8, linestyle="--")
        ax2.axhline(0.5,  color="#E53935", linewidth=0.8, linestyle=":", alpha=0.5)
        ax2.axhline(-0.5, color="#E53935", linewidth=0.8, linestyle=":", alpha=0.5)
        ax2.axhline(0.3,  color="#FB8C00", linewidth=0.8, linestyle=":", alpha=0.5)
        ax2.axhline(-0.3, color="#FB8C00", linewidth=0.8, linestyle=":", alpha=0.5)
        ax2.set_xticks(range(LATENT_DIM))
        ax2.set_xticklabels([f"lat_{d}" for d in range(LATENT_DIM)], fontsize=8)
        ax2.set_title(f"{task.upper()} — Cohen's d (Unstable − Stable)", fontsize=10)
        ax2.set_ylabel("Cohen's d")
        ax2.spines[["top", "right"]].set_visible(False)

        # Etichetta soglie
        ax2.text(LATENT_DIM - 0.5, 0.52,  "large", fontsize=7, color="#E53935", ha="right")
        ax2.text(LATENT_DIM - 0.5, 0.32,  "medium", fontsize=7, color="#FB8C00", ha="right")

    fig.suptitle(
        f"Latent Trend Analysis — {ARCH_MODE}\n"
        "Left: slope distributions (blue=stable, red=unstable) | "
        "Right: effect size of trend difference",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out_path = os.path.join(RESULTS_PATH, f"trend_analysis_{ARCH_MODE}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAnalysis plot saved -> {out_path}")

    # ── SUMMARY TESTUALE ─────────────────────────────────────────────────────
    print("\n── Dimensioni con effetto medio-grande (|d| > 0.3) ──")
    for task in tasks:
        sub = df[df["task"] == task]
        stable   = sub[sub["label"] == 0]
        unstable = sub[sub["label"] == 1]
        useful = []
        for d, col in enumerate(slope_cols):
            s_v = stable[col].dropna().values
            u_v = unstable[col].dropna().values
            if len(s_v) > 1 and len(u_v) > 1:
                pooled = np.sqrt((np.var(s_v, ddof=1)*(len(s_v)-1) +
                                  np.var(u_v, ddof=1)*(len(u_v)-1)) /
                                 (len(s_v)+len(u_v)-2)) + 1e-8
                cd = abs((np.mean(u_v) - np.mean(s_v)) / pooled)
                if cd > 0.3:
                    useful.append(f"lat_{d} (d={cd:.2f})")
        print(f"  {task}: {useful if useful else 'nessuna'}")

    verdict = (
        "→ VALE LA PENA aggiungere la slope come feature."
        if any(
            abs((df[df["label"]==1][c].mean() - df[df["label"]==0][c].mean()) /
                (df[c].std() + 1e-8)) > 0.3
            for c in slope_cols
        )
        else "→ Il trend non sembra sufficientemente discriminativo. Valuta se aggiungerlo."
    )
    print(f"\n{verdict}")

if __name__ == "__main__":
    main()