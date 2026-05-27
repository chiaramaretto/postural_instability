import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.preprocessing import StandardScaler

CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH    = "posturalInstability/cnn_gru/results/"
ARCH_MODE       = "autoencoder"  # "autoencoder" or "classifier"_
LATENT_DIM      = 8

# ═════════════════════════════════════════════
# 1. SETUP & CLASSIFIER DICTIONARY
# ═════════════════════════════════════════════

def split_lat_hc(X):
    # Stance + walking: latent mean/std/max for stance and walking blocks
    lat_block = 2 * (4 * LATENT_DIM)  # 64 when LATENT_DIM = 8
    return X[:, :lat_block], X[:, lat_block:]

def apply_variance_threshold(Xtr, Xte):
    v = VarianceThreshold(threshold=1e-6)
    Xtr_f = v.fit_transform(Xtr)
    Xte_f = v.transform(Xte)
    return Xtr_f, Xte_f


def apply_z_scaling(Xtr, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    return Xtr_s, Xte_s

def get_classifiers():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=80, max_depth=6, min_samples_leaf=3, 
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=42
        ),
        "SVM": SVC(
            kernel='rbf', probability=True, class_weight='balanced', random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=42
        )
    }

# ═════════════════════════════════════════════
# 2. EVALUATION PIPELINE
# ═════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    train_path = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{ARCH_MODE}.csv")
    test_path = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{ARCH_MODE}.csv")

    print(f"Loading enriched features from disk ({ARCH_MODE})...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Extract metadata targets
    y_fit = train_df["y_true"].values
    y_test = test_df["y_true"].values
    
    # Recuperiamo in sicurezza la colonna a 4 classi se presente nel CSV
    y_test_4cls = test_df["y_true_4cls"].values if "y_true_4cls" in test_df.columns else None
    
    dsets_test = test_df["dataset"].values
    sids_test = test_df["subjectID"].values

    # Drop metadata to get raw feature matrices
    feat_cols = [c for c in train_df.columns if c.startswith("Feat_")]
    X_fit = train_df[feat_cols].values
    X_test = test_df[feat_cols].values

    # Split into Ablation Subsets
    X_fit_lat,  X_fit_hc  = split_lat_hc(X_fit)
    X_test_lat, X_test_hc = split_lat_hc(X_test)

    # Filter zero-variance features
    X_fit_lat_f,  X_test_lat_f  = apply_variance_threshold(X_fit_lat,  X_test_lat)
    X_fit_hc_f,   X_test_hc_f   = apply_variance_threshold(X_fit_hc,   X_test_hc)
    X_fit_full_f, X_test_full_f = apply_variance_threshold(X_fit,   X_test)

    # Z-score scaling fitted on train only and then applied to test
    X_fit_lat_s,  X_test_lat_s  = apply_z_scaling(X_fit_lat_f,  X_test_lat_f)
    X_fit_hc_s,   X_test_hc_s   = apply_z_scaling(X_fit_hc_f,   X_test_hc_f)
    X_fit_full_s, X_test_full_s = apply_z_scaling(X_fit_full_f, X_test_full_f)

    # --- FEATURE SELECTION: RECURSIVE FEATURE ELIMINATION (RFE) ---
    print("\nPerforming Recursive Feature Elimination (RFE) on Combined set...")
    rfe_estimator = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=30, step=5)
    X_fit_rfe = rfe.fit_transform(X_fit_full_s, y_fit)
    X_test_rfe = rfe.transform(X_test_full_s)
    print(f"RFE selected {X_fit_rfe.shape[1]} features out of {X_fit_full_s.shape[1]}")

    feature_sets = {
        "Handcrafted":  (X_fit_hc_s, X_test_hc_s),
        "Latent":       (X_fit_lat_s, X_test_lat_s),
        "Combined":     (X_fit_full_s, X_test_full_s),
        "Combined_RFE": (X_fit_rfe, X_test_rfe)
    }

    classifiers = get_classifiers()
    
    # Trackers for saving
    summary_results = []
    dataset_results = []
    
    # Initialize the prediction dataframe with ground truth
    pred_df = pd.DataFrame({
        "subjectID": sids_test,
        "dataset": dsets_test,
        "y_true": y_test.astype(int)
    })
    
    # Aggiungiamo la colonna a 4 classi nel DataFrame finale delle predizioni
    if y_test_4cls is not None:
        pred_df["y_true_4cls"] = y_test_4cls.astype(int)

    print("\n" + "═" * 60)
    print(" STARTING DOUBLE ABLATION STUDY (WITH RFE)")
    print("═" * 60)

    for clf_name, clf in classifiers.items():
        for feat_name, (X_tr, X_te) in feature_sets.items():
            print(f"Training {clf_name} on {feat_name} features ({X_tr.shape[1]} dims)...")
            
            clf.fit(X_tr, y_fit)
            
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te)
            
            # Save predictions iteratively
            col_suffix = f"{clf_name}_{feat_name}"
            pred_df[f"y_pred_{col_suffix}"] = y_pred.astype(int)
            pred_df[f"prob_{col_suffix}"] = y_prob

            # Calculate metrics
            summary_results.append({
                "Classifier": clf_name,
                "Feature Set": feat_name,
                "N Features": X_tr.shape[1],
                "Train Acc": round(accuracy_score(y_fit, clf.predict(X_tr)), 4),
                "Test Acc": round(accuracy_score(y_test, y_pred), 4),
                "Balanced Acc": round(balanced_accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "Macro F1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4) if len(np.unique(y_test)) > 1 else float("nan")
            })

            # Per-dataset test report: keeps the 3 datasets visible in the final analysis.
            for dataset_name in pd.unique(dsets_test):
                ds_mask = dsets_test == dataset_name
                if ds_mask.sum() == 0:
                    continue

                ds_y_true = y_test[ds_mask]
                ds_y_pred = y_pred[ds_mask]
                ds_prob = y_prob[ds_mask]

                dataset_results.append({
                    "Classifier": clf_name,
                    "Feature Set": feat_name,
                    "Dataset": dataset_name,
                    "Test Samples": int(ds_mask.sum()),
                    "Accuracy": round(accuracy_score(ds_y_true, ds_y_pred), 4),
                    "Balanced Acc": round(balanced_accuracy_score(ds_y_true, ds_y_pred), 4),
                    "Precision": round(precision_score(ds_y_true, ds_y_pred, zero_division=0), 4),
                    "Recall": round(recall_score(ds_y_true, ds_y_pred, zero_division=0), 4),
                    "Macro F1": round(f1_score(ds_y_true, ds_y_pred, average="macro", zero_division=0), 4),
                    "ROC-AUC": round(roc_auc_score(ds_y_true, ds_prob), 4) if len(np.unique(ds_y_true)) > 1 else float("nan"),
                })

    # ═════════════════════════════════════════════
    # 3. SAVE RESULTS
    # ═════════════════════════════════════════════

    summary_df = pd.DataFrame(summary_results)
    dataset_df = pd.DataFrame(dataset_results)
    
    print("\n" + "═" * 60)
    print(" SUMMARY RESULTS")
    print("═" * 60)
    print(summary_df.to_string(index=False))

    print("\n" + "═" * 60)
    print(" DATASET-WISE TEST RESULTS")
    print("═" * 60)
    print(dataset_df.to_string(index=False))

    # Export metrics
    summary_path = os.path.join(RESULTS_PATH, f"double_ablation_metrics_{ARCH_MODE}.csv")
    summary_df.to_csv(summary_path, index=False)

    dataset_summary_path = os.path.join(RESULTS_PATH, f"double_ablation_metrics_by_dataset_{ARCH_MODE}.csv")
    dataset_df.to_csv(dataset_summary_path, index=False)
    
    # Export predictions for further KDE/Misclassification plotting
    preds_path = os.path.join(RESULTS_PATH, f"double_ablation_predictions_{ARCH_MODE}.xlsx")
    pred_df.to_excel(preds_path, index=False)

    print(f"\nAll metrics saved to -> {summary_path}")
    print(f"Dataset-wise metrics saved to -> {dataset_summary_path}")
    print(f"All predictions saved to -> {preds_path}")

if __name__ == "__main__":
    main()