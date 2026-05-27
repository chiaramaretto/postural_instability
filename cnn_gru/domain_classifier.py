import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH    = "posturalInstability/cnn_gru/results/"
FEATURE_PREFIX = "train_features_enriched_"
ARCH_MODE = "autoencoder"  # "autoencoder", "classifier", "classifier_mmd"
def discover_feature_modes(checkpoint_path=CHECKPOINT_PATH):
    modes = []
    for fname in os.listdir(checkpoint_path):
        if not fname.startswith(FEATURE_PREFIX) or not fname.endswith('.csv'):
            continue
        mode = fname[len(FEATURE_PREFIX):-4]
        # require both train and test files to exist for this mode
        train_path = os.path.join(checkpoint_path, f"train_features_enriched_{mode}.csv")
        test_path = os.path.join(checkpoint_path, f"test_features_enriched_{mode}.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            modes.append(mode)
    # sort deterministically, prefer known order if present
    preferred = ["autoencoder", "classifier", "classifier_mmd"]
    modes_sorted = [m for m in preferred if m in modes] + sorted([m for m in modes if m not in preferred])
    return modes_sorted


def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": __import__("sklearn.ensemble", fromlist=["GradientBoostingClassifier"]).GradientBoostingClassifier(
            n_estimators=120,
            max_depth=4,
            random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        ),
    }


def load_feature_split(mode):
    train_path = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{mode}.csv")
    test_path = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{mode}.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing enriched feature files for mode '{mode}'.")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def get_feature_matrix(df):
    feat_cols = [c for c in df.columns if c.startswith("Feat_")]
    if not feat_cols:
        raise ValueError("No feature columns found. Expected columns starting with 'Feat_'.")
    return df[feat_cols].values


def clean_features(x_train, x_test):
    selector = VarianceThreshold(threshold=1e-6)
    x_train_f = selector.fit_transform(x_train)
    x_test_f = selector.transform(x_test)
    return x_train_f, x_test_f


def train_domain_model(model, x_train, y_train):
    return make_pipeline(StandardScaler(), model).fit(x_train, y_train)


def evaluate_combo(mode, model_name, model):
    train_df, test_df = load_feature_split(mode)

    x_train = get_feature_matrix(train_df)
    x_test = get_feature_matrix(test_df)
    y_train_raw = train_df["dataset"].values
    y_test_raw = test_df["dataset"].values

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    x_train_f, x_test_f = clean_features(x_train, x_test)
    clf = train_domain_model(model, x_train_f, y_train)

    y_pred = clf.predict(x_test_f)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    chance = 1.0 / len(encoder.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(encoder.classes_)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    return {
        "Feature Set": mode,
        "Model": model_name,
        "Train Samples": len(train_df),
        "Test Samples": len(test_df),
        "Raw Features": x_train.shape[1],
        "Kept Features": x_train_f.shape[1],
        "Domains": len(encoder.classes_),
        "Chance": chance,
        "Accuracy": acc,
        "Balanced Acc": bal_acc,
        "Macro F1": macro_f1,
        "Gap vs Chance": acc - chance,
        "Classes": encoder.classes_,
        "Confusion Matrix": cm,
        "Confusion Matrix Norm": cm_norm,
    }

def main(arch_mode=None):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    print("Caricamento delle feature e benchmark dei modelli...\n")

    models = get_models()
    results = []

    feature_modes = discover_feature_modes(CHECKPOINT_PATH)
    if not feature_modes:
        print("No enriched feature CSV pairs found in models folder. Exiting.")
        return

    # determine which feature modes to run based on ARCH_MODE or CLI arg
    active_modes = feature_modes
    chosen = arch_mode or ARCH_MODE
    if chosen:
        # allow comma-separated list or single value; 'all' means run everything
        if isinstance(chosen, str):
            parts = [p.strip() for p in chosen.split(",") if p.strip()]
        else:
            parts = list(chosen)
        # build selected modes with some convenient shorthands
        selected = []
        for p in parts:
            pl = p.lower()
            if pl == "all":
                selected = feature_modes
                break
            if pl == "mmd":
                for m in feature_modes:
                    if m.endswith("_mmd") and m not in selected:
                        selected.append(m)
                continue
            # exact match or prefix match (e.g., 'classifier' -> 'classifier' and 'classifier_mmd')
            matched_any = False
            for m in feature_modes:
                if m == p or m.lower().startswith(pl):
                    if m not in selected:
                        selected.append(m)
                    matched_any = True
            if not matched_any:
                # record as missing for warning later
                selected.append(f"__MISSING__::{p}")

        # separate missing and valid
        missing = [s.split("::", 1)[1] for s in selected if s.startswith("__MISSING__::")]
        active_modes = [s for s in selected if not s.startswith("__MISSING__::")]
        if missing:
            print(f"Warning: requested ARCH_MODE entries not found: {missing}")
        if not active_modes:
            print("No matching feature modes selected. Exiting.")
            return

    for mode in active_modes:
        print(f"Feature set: {mode}")
        for model_name, model in models.items():
            print(f"  -> {model_name}")
            result = evaluate_combo(mode, model_name, model)
            results.append(result)

    summary_df = pd.DataFrame([
        {
            "Feature Set": r["Feature Set"],
            "Model": r["Model"],
            "Train Samples": r["Train Samples"],
            "Test Samples": r["Test Samples"],
            "Raw Features": r["Raw Features"],
            "Kept Features": r["Kept Features"],
            "Domains": r["Domains"],
            "Chance": round(r["Chance"], 4),
            "Accuracy": round(r["Accuracy"], 4),
            "Balanced Acc": round(r["Balanced Acc"], 4),
            "Macro F1": round(r["Macro F1"], 4),
            "Gap vs Chance": round(r["Gap vs Chance"], 4),
        }
        for r in results
    ])

    summary_path = os.path.join(RESULTS_PATH, "domain_classifier_benchmark.csv")
    summary_df.to_csv(summary_path, index=False)

    pivot = summary_df.pivot(index="Feature Set", columns="Model", values="Accuracy").reindex(active_modes)
    chance_by_mode = summary_df.groupby("Feature Set")["Chance"].first().reindex(active_modes)

    plt.figure(figsize=(11, 6))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Dataset classification accuracy"},
    )
    for idx, mode in enumerate(pivot.index):
        ax.hlines(idx, *ax.get_xlim(), colors="white", linewidth=0.5)
    plt.title("How easily each feature set reveals the dataset/domain", fontsize=13, pad=12)
    plt.xlabel("Classifier")
    plt.ylabel("Feature set")
    plt.tight_layout()
    overview_path = os.path.join(RESULTS_PATH, "domain_classifier_heatmap.png")
    plt.savefig(overview_path, dpi=160)
    plt.close()

    # Compare MMD vs non-MMD for same base architectures and save comparison CSVs
    comps_saved = []
    bases = sorted({m.replace("_mmd", "") for m in active_modes})
    for base in bases:
        non = base
        mmd = base + "_mmd"
        if non in active_modes and mmd in active_modes:
            comp_rows = []
            for model_name in summary_df["Model"].unique():
                row_non = summary_df[(summary_df["Feature Set"] == non) & (summary_df["Model"] == model_name)]
                row_mmd = summary_df[(summary_df["Feature Set"] == mmd) & (summary_df["Model"] == model_name)]
                if row_non.empty or row_mmd.empty:
                    continue
                rn = row_non.iloc[0]
                rm = row_mmd.iloc[0]
                comp = {
                    "Base": base,
                    "Model": model_name,
                    "Accuracy_nonmmd": rn["Accuracy"],
                    "Accuracy_mmd": rm["Accuracy"],
                    "Accuracy_diff": rm["Accuracy"] - rn["Accuracy"],
                    "BalancedAcc_nonmmd": rn["Balanced Acc"],
                    "BalancedAcc_mmd": rm["Balanced Acc"],
                    "BalancedAcc_diff": rm["Balanced Acc"] - rn["Balanced Acc"],
                    "MacroF1_nonmmd": rn["Macro F1"],
                    "MacroF1_mmd": rm["Macro F1"],
                    "MacroF1_diff": rm["Macro F1"] - rn["Macro F1"],
                    "KeptFeatures_nonmmd": rn["Kept Features"],
                    "KeptFeatures_mmd": rm["Kept Features"],
                    "KeptFeatures_diff": rm["Kept Features"] - rn["Kept Features"],
                }
                comp_rows.append(comp)

            if comp_rows:
                comp_df = pd.DataFrame(comp_rows)
                comp_path = os.path.join(RESULTS_PATH, f"domain_shift_comparison_{base}.csv")
                comp_df.to_csv(comp_path, index=False)
                comps_saved.append(comp_path)
                print(f"\nDomain-shift comparison saved -> {comp_path}")
                for _, r in comp_df.iterrows():
                    print(f"  {r['Model']}: Accuracy diff={r['Accuracy_diff']:.4f}, BalancedAcc diff={r['BalancedAcc_diff']:.4f}, MacroF1 diff={r['MacroF1_diff']:.4f}")

    print("\nSummary table:")
    print(summary_df.sort_values(["Feature Set", "Accuracy"], ascending=[True, False]).to_string(index=False))

    print("\nBest model per feature set:")
    for mode in active_modes:
        sub = summary_df[summary_df["Feature Set"] == mode].sort_values("Accuracy", ascending=False).iloc[0]
        print(
            f"  {mode}: {sub['Model']} | acc={sub['Accuracy']:.2%} | chance={chance_by_mode[mode]:.2%} | gap={sub['Gap vs Chance']:.2%}"
        )

    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved heatmap -> {overview_path}")

    strongest = summary_df.sort_values("Accuracy", ascending=False).iloc[0]
    weakest = summary_df.sort_values("Accuracy", ascending=True).iloc[0]
    print(
        "\nInterpretation: if the dataset is easy to classify from the extracted features, the features still carry domain information. "
        f"In this run, the strongest separation is {strongest['Feature Set']} + {strongest['Model']} ({strongest['Accuracy']:.2%}), "
        f"while the weakest is {weakest['Feature Set']} + {weakest['Model']} ({weakest['Accuracy']:.2%}). "
        "A lower accuracy for `classifier_mmd` than for `classifier` or `autoencoder` would suggest that MMD reduces the domain shift."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark domain classifiers on enriched feature sets")
    parser.add_argument("--arch-mode", dest="arch_mode", help="Comma-separated feature modes to run (or 'all'). Overrides ARCH_MODE env var.")
    args = parser.parse_args()
    try:
        main(arch_mode=args.arch_mode)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)