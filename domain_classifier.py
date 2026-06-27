import os
import argparse
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

CHECKPOINT_PATH = "posturalInstability/models/"
RESULTS_PATH    = "posturalInstability/results/"
LATENT_DIM      = 8
LATENT_BLOCK    = 4 * LATENT_DIM   # 32 dims
RANDOM_STATE    = 42

# DA variants to compare
DA_VARIANTS = ["baseline", "coral", "mmd"]

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ══════════════════════════════════════════════════════════════════════════════
# Discovery
# ══════════════════════════════════════════════════════════════════════════════

def discover_arch_modes(checkpoint_path=CHECKPOINT_PATH):
    arch_modes = set()
    for da in DA_VARIANTS:
        for fname in os.listdir(checkpoint_path):
            prefix = f"train_features_enriched_"
            suffix = f"_{da}.csv"
            if fname.startswith(prefix) and fname.endswith(suffix):
                arch = fname[len(prefix):-len(suffix)]
                # check all three variants exist
                all_exist = all(
                    os.path.exists(os.path.join(
                        checkpoint_path,
                        f"train_features_enriched_{arch}_{v}.csv")) and
                    os.path.exists(os.path.join(
                        checkpoint_path,
                        f"test_features_enriched_{arch}_{v}.csv"))
                    for v in DA_VARIANTS
                )
                if all_exist:
                    arch_modes.add(arch)
    return sorted(arch_modes)


# ══════════════════════════════════════════════════════════════════════════════
# Feature loading & splitting
# ══════════════════════════════════════════════════════════════════════════════

def load_variant(arch_mode, da_variant, seed=RANDOM_STATE, checkpoint_path=CHECKPOINT_PATH):
    tag = f"{arch_mode}_{da_variant}_seed{seed}"
    tr  = os.path.join(checkpoint_path, f"train_features_enriched_{tag}.csv")
    te  = os.path.join(checkpoint_path, f"test_features_enriched_{tag}.csv")
    if not os.path.exists(tr) or not os.path.exists(te):
        raise FileNotFoundError(f"Missing: {tr} or {te}")
    return pd.read_csv(tr), pd.read_csv(te)


def get_feature_blocks(df):
    """Return dict of feature blocks: Latent, Handcrafted, Combined."""
    feat_cols = [c for c in df.columns if c.startswith("Feat_")]
    # filter NaN rows
    valid     = df[feat_cols].notna().all(axis=1)
    df_v      = df[valid]
    X         = df_v[feat_cols].values
    X_lat     = X[:, :LATENT_BLOCK]
    X_hc      = X[:, LATENT_BLOCK:]
    return {
        "Latent":      (X_lat, df_v),
        "Handcrafted": (X_hc,  df_v),
        "Combined":    (X,     df_v),
    }, valid


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

def get_domain_models(seed=RANDOM_STATE):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=4, random_state=seed,
        ),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=seed,
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=seed,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_domain(X_train, y_train, X_test, y_test,
                    class_names, model_name, model):
    """Train domain classifier, return metric dict."""
    clf = make_pipeline(
        VarianceThreshold(threshold=1e-6),
        StandardScaler(),
        model,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    n_cls  = len(np.unique(y_train))
    chance = 1.0 / n_cls
    auc    = (roc_auc_score(y_test, y_prob[:, 1])
              if n_cls == 2
              else roc_auc_score(y_test, y_prob,
                                 average="macro", multi_class="ovr"))

    return {
        "Model":        model_name,
        "Accuracy":     round(accuracy_score(y_test, y_pred),    4),
        "Balanced Acc": round(balanced_accuracy_score(y_test, y_pred), 4),
        "Macro F1":     round(f1_score(y_test, y_pred, average="macro",
                                        zero_division=0), 4),
        "AUC":          round(auc, 4),
        "Chance":       round(chance, 4),
        "Gap":          round(accuracy_score(y_test, y_pred) - chance, 4),
    }


def run_arch(arch_mode, seed=RANDOM_STATE):
    set_seeds(seed)
    models  = get_domain_models(seed=seed)
    records = []

    for da_variant in DA_VARIANTS:
        print(f"\n  DA variant: {da_variant.upper()}")
        try:
            tr_df, te_df = load_variant(arch_mode, da_variant, seed=seed)
        except FileNotFoundError as e:
            print(f"    [!] {e}")
            continue

        # Encode domain labels
        enc = LabelEncoder()
        enc.fit(np.concatenate([tr_df["dataset"].values,
                                te_df["dataset"].values]))
        class_names = enc.classes_
        n_domains   = len(class_names)
        print(f"    Domains ({n_domains}): {list(class_names)}")

        feat_blocks_tr, valid_tr = get_feature_blocks(tr_df)
        feat_blocks_te, valid_te = get_feature_blocks(te_df)

        for feat_name in ["Latent", "Handcrafted", "Combined"]:
            X_tr_raw, df_tr_v = feat_blocks_tr[feat_name]
            X_te_raw, df_te_v = feat_blocks_te[feat_name]

            if X_tr_raw.shape[1] == 0:
                continue

            y_tr = enc.transform(df_tr_v["dataset"].values)
            y_te = enc.transform(df_te_v["dataset"].values)

            for model_name, model in models.items():
                m = evaluate_domain(X_tr_raw, y_tr, X_te_raw, y_te,
                                    class_names, model_name, model)
                m.update({
                    "Arch Mode":    arch_mode,
                    "DA Variant":   da_variant,
                    "Feature Block": feat_name,
                    "N Domains":    n_domains,
                    "N Train":      len(X_tr_raw),
                    "N Test":       len(X_te_raw),
                })
                records.append(m)
                print(f"    {feat_name:<12} {model_name:<20} "
                      f"Acc={m['Accuracy']:.3f}  AUC={m['AUC']:.3f}  "
                      f"Gap={m['Gap']:+.3f}")

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table: baseline vs coral vs mmd
# ══════════════════════════════════════════════════════════════════════════════

def build_comparison(df):
    """
    For each (arch_mode, feature_block, model), compute delta metrics
    relative to baseline.
    """
    rows = []
    for (arch, feat, model), grp in df.groupby(["Arch Mode", "Feature Block", "Model"]):
        base = grp[grp["DA Variant"] == "baseline"]
        if base.empty:
            continue
        b = base.iloc[0]
        for da in ["coral", "mmd"]:
            var = grp[grp["DA Variant"] == da]
            if var.empty:
                continue
            v = var.iloc[0]
            rows.append({
                "Arch Mode":     arch,
                "Feature Block": feat,
                "Model":         model,
                "DA Variant":    da,
                # absolute values
                "Acc_baseline":  b["Accuracy"],
                "Acc_da":        v["Accuracy"],
                "AUC_baseline":  b["AUC"],
                "AUC_da":        v["AUC"],
                "BalAcc_baseline": b["Balanced Acc"],
                "BalAcc_da":     v["Balanced Acc"],
                # deltas (negative = more domain-invariant = better)
                "Acc_delta":     round(v["Accuracy"]     - b["Accuracy"],     4),
                "AUC_delta":     round(v["AUC"]          - b["AUC"],          4),
                "BalAcc_delta":  round(v["Balanced Acc"] - b["Balanced Acc"], 4),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(df, arch_mode):
    """Accuracy heatmap: rows = DA variant × feature block, cols = model."""
    df_arch = df[df["Arch Mode"] == arch_mode].copy()
    df_arch["Row"] = df_arch["DA Variant"] + " | " + df_arch["Feature Block"]
    pivot = df_arch.pivot_table(
        index="Row", columns="Model", values="Accuracy", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.55 + 2)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Reds",
                vmin=0.0, vmax=1.0, linewidths=0.4,
                cbar_kws={"label": "Domain classification accuracy"},
                ax=ax)
    ax.set_title(f"Domain separability — {arch_mode}\n"
                 f"(lower = more domain-invariant)", fontsize=11)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("DA variant | Feature block")
    plt.tight_layout()
    out = os.path.join(RESULTS_PATH,
                       f"domain_classifier_heatmap_{arch_mode}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved -> {out}")


def plot_delta(comp_df, arch_mode):
    """Delta bar chart: how much does each DA method reduce domain accuracy."""
    sub = comp_df[comp_df["Arch Mode"] == arch_mode].copy()
    if sub.empty:
        return

    # Average delta across models, per (DA Variant, Feature Block)
    agg = sub.groupby(["DA Variant", "Feature Block"])["Acc_delta"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(agg["Feature Block"].unique()))
    width = 0.35
    feats = sorted(agg["Feature Block"].unique())
    colors = {"coral": "#DD8452", "mmd": "#4C72B0"}

    for i, da in enumerate(["coral", "mmd"]):
        vals = [agg[(agg["DA Variant"] == da) & (agg["Feature Block"] == f)]["Acc_delta"].values
                for f in feats]
        vals = [v[0] if len(v) > 0 else 0.0 for v in vals]
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=da.upper(), color=colors[da], alpha=0.85)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(feats)
    ax.set_ylabel("Δ Accuracy vs baseline\n(negative = more domain-invariant)")
    ax.set_title(f"DA effectiveness — {arch_mode}")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_PATH,
                       f"domain_classifier_delta_{arch_mode}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Delta plot saved -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(arch_mode_filter=None, seed=RANDOM_STATE):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    arch_modes = ["autoencoder", "classifier"] if not arch_mode_filter else [arch_mode_filter]

    all_records = []
    for arch in arch_modes:
        print(f"\n{'═'*60}")
        print(f"  ARCH MODE: {arch.upper()}  SEED: {seed}")
        print(f"{'═'*60}")
        try:
            all_records.extend(run_arch(arch, seed=seed))
        except FileNotFoundError as e:
            print(f"  [!] Skipping {arch}: {e}")

    if not all_records:
        print("No results produced.")
        return

    results_df  = pd.DataFrame(all_records)
    comp_df     = build_comparison(results_df)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    results_path = os.path.join(RESULTS_PATH, f"domain_classifier_results_seed{seed}.csv")
    comp_path    = os.path.join(RESULTS_PATH, f"domain_classifier_comparison_seed{seed}.csv")
    results_df.to_csv(results_path, index=False)
    comp_df.to_csv(comp_path,       index=False)
    print(f"\nResults saved     -> {results_path}")
    print(f"Comparison saved  -> {comp_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    for arch in arch_modes:
        plot_heatmap(results_df, arch)
        plot_delta(comp_df, arch)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(" SUMMARY: mean accuracy across models (lower domain acc = better DA)")
    print(f"{'═'*70}")
    summary = (
        results_df.groupby(["Arch Mode", "DA Variant", "Feature Block"])
        ["Accuracy"].mean().round(4).reset_index()
    )
    for arch in arch_modes:
        print(f"\n  {arch}:")
        sub = summary[summary["Arch Mode"] == arch]
        print(sub[["DA Variant","Feature Block","Accuracy"]].to_string(index=False))

    if not comp_df.empty:
        print(f"\n{'─'*70}")
        print(" DELTA vs baseline (negative = domain confusion increased = better)")
        print(f"{'─'*70}")
        delta_summary = (
            comp_df.groupby(["Arch Mode","DA Variant","Feature Block"])
            [["Acc_delta","AUC_delta"]].mean().round(4).reset_index()
        )
        print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Domain classifier: baseline vs CORAL vs MMD")
    parser.add_argument("--arch-mode", default=None,
                        help="Filter to a single arch mode (e.g. 'autoencoder')")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    main(arch_mode_filter=args.arch_mode, seed=args.seed)
