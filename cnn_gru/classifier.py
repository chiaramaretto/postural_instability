import os
import argparse
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
LATENT_DIM      = 8
FEATURE_PREFIX  = "train_features_enriched_"

# ═════════════════════════════════════════════
# 1. SETUP & CLASSIFIER DICTIONARY
# ═════════════════════════════════════════════

def split_lat_hc(X):
    lat_block = (4 * LATENT_DIM)  
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
            n_estimators=50, max_depth=3, min_samples_leaf=3, 
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        ),
        "SVM": SVC(
            kernel='rbf', probability=True, class_weight='balanced', random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=42
        )
    }

def discover_feature_mode(checkpoint_path=CHECKPOINT_PATH):
    available_modes = []
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Directory {checkpoint_path} non trovata.")

    for fname in os.listdir(checkpoint_path):
        if fname.startswith(FEATURE_PREFIX) and fname.endswith(".csv"):
            mode = fname[len(FEATURE_PREFIX):-4]
            available_modes.append(mode)

    if not available_modes:
        raise FileNotFoundError("Nessun file train_features_enriched_*.csv trovato.")
    
    # Ritorna il primo trovato se è in modalità auto
    return sorted(available_modes)[0]

def binary_confusion_stats(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    hc_recall = tn / (tn + fp + 1e-8)
    pd_recall = tp / (tp + fn + 1e-8)
    return tn, fp, fn, tp, hc_recall, pd_recall

# ═════════════════════════════════════════════
# 2. EVALUATION PIPELINE
# ═════════════════════════════════════════════

def run_ablation(arch_mode):

    suffix = ["baseline", "coral", "mmd"]
    for s in suffix:
        train_path = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{arch_mode}_{s}.csv")
        test_path = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{arch_mode}_{s}.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"File mancanti per la modalità: '{arch_mode}'")

        print(f"Loading enriched features from disk ({arch_mode})...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Extract metadata targets
        y_fit = train_df["y_true"].values
        y_test = test_df["y_true"].values
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
        rfe_estimator = RandomForestClassifier(n_estimators=80, max_depth=3, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=15, step=5)
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
        
        pred_df = pd.DataFrame({"subjectID": sids_test, "dataset": dsets_test, "y_true": y_test.astype(int)})
        if y_test_4cls is not None:
            pred_df["y_true_4cls"] = y_test_4cls.astype(int)

        print("\n" + "═" * 60)
        print(f" STARTING DOUBLE ABLATION STUDY ({arch_mode.upper()})")
        print("═" * 60)

        for clf_name, clf in classifiers.items():
            for feat_name, (X_tr, X_te) in feature_sets.items():
                print(f"Training {clf_name} on {feat_name} features ({X_tr.shape[1]} dims)...")
                
                clf.fit(X_tr, y_fit)
                
                y_pred = clf.predict(X_te)
                y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te)
                
                col_suffix = f"{clf_name}_{feat_name}"
                pred_df[f"y_pred_{col_suffix}"] = y_pred.astype(int)
                pred_df[f"prob_{col_suffix}"] = y_prob

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

                # Per-dataset test report
                for dataset_name in pd.unique(dsets_test):
                    ds_mask = dsets_test == dataset_name
                    if ds_mask.sum() == 0: continue

                    ds_y_true = y_test[ds_mask]
                    ds_y_pred = y_pred[ds_mask]
                    tn, fp, fn, tp, hc_recall, pd_recall = binary_confusion_stats(ds_y_true, ds_y_pred)

                    dataset_results.append({
                        "Classifier": clf_name, "Feature Set": feat_name, "Dataset": dataset_name,
                        "Test Samples": int(ds_mask.sum()), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                        "HC Recall": round(hc_recall, 4), "PD Recall": round(pd_recall, 4),
                        "Balanced Acc": round(balanced_accuracy_score(ds_y_true, ds_y_pred), 4),
                        "Macro F1": round(f1_score(ds_y_true, ds_y_pred, average="macro", zero_division=0), 4),
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

        summary_path = os.path.join(RESULTS_PATH, f"double_ablation_metrics_{arch_mode}.csv")
        dataset_summary_path = os.path.join(RESULTS_PATH, f"double_ablation_metrics_by_dataset_{arch_mode}.csv")
        preds_path = os.path.join(RESULTS_PATH, f"double_ablation_predictions_{arch_mode}.xlsx")
        
        summary_df.to_csv(summary_path, index=False)
        dataset_df.to_csv(dataset_summary_path, index=False)
        pred_df.to_excel(preds_path, index=False)

        print(f"\nAll metrics saved to -> {summary_path}")
        print(f"Dataset-wise metrics saved to -> {dataset_summary_path}")
        print(f"All predictions saved to -> {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classifier ablation on enriched features")
    parser.add_argument("--arch-mode", default="auto", help="Feature mode to load, or 'auto' to discover automatically.")
    args = parser.parse_args()

    os.makedirs(RESULTS_PATH, exist_ok=True)

    if args.arch_mode == "auto":
        print("Auto-discovering feature mode...")
        target_mode = discover_feature_mode(CHECKPOINT_PATH)
    else:
        target_mode = args.arch_mode

    run_ablation(target_mode)