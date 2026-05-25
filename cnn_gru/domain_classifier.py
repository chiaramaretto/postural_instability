import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH    = "posturalInstability/cnn_gru/results/"
FEATURE_MODES   = ["autoencoder", "classifier", "classifier_mmd"]


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

def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)

    print("Caricamento delle feature e benchmark dei modelli...\n")

    models = get_models()
    results = []

    for mode in FEATURE_MODES:
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

    pivot = summary_df.pivot(index="Feature Set", columns="Model", values="Accuracy").reindex(FEATURE_MODES)
    chance_by_mode = summary_df.groupby("Feature Set")["Chance"].first().reindex(FEATURE_MODES)

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

    print("\nSummary table:")
    print(summary_df.sort_values(["Feature Set", "Accuracy"], ascending=[True, False]).to_string(index=False))

    print("\nBest model per feature set:")
    for mode in FEATURE_MODES:
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
    main()