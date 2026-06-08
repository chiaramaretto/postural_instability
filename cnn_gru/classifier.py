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
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
)
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH    = "posturalInstability/cnn_gru/results/"
FEATURE_PREFIX  = "train_features_enriched_"
LATENT_DIM      = 8

# ═════════════════════════════════════════════
# 1. UTILS
# ═════════════════════════════════════════════

def split_lat_hc(X):
    # Il blocco latente ora è 32 (4 stat * 8 dim)
    lat_block = (4 * LATENT_DIM)  
    return X[:, :lat_block], X[:, lat_block:]

def get_classifiers():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "SVM": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)
    }

def discover_variants(base_mode, checkpoint_path=CHECKPOINT_PATH):
    """Trova tutte le varianti (es: autoencoder_original, autoencoder_coral)"""
    found = []
    for fname in os.listdir(checkpoint_path):
        if fname.startswith(f"{FEATURE_PREFIX}{base_mode}") and fname.endswith(".csv"):
            mode_variant = fname[len(FEATURE_PREFIX):-4]
            # Verifica che esista la coppia train/test
            if os.path.exists(os.path.join(checkpoint_path, f"test_features_enriched_{mode_variant}.csv")):
                found.append(mode_variant)
    return sorted(found)

# ═════════════════════════════════════════════
# 2. EVALUATION PIPELINE
# ═════════════════════════════════════════════

def run_ablation(mode_variant):
    print(f"\n{'═'*80}\n>>> Esecuzione Ablazione: {mode_variant}\n{'═'*80}")
    
    train_path = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{mode_variant}.csv")
    test_path = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{mode_variant}.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_fit, y_test = train_df["y_true"].values, test_df["y_true"].values
    feat_cols = [c for c in train_df.columns if c.startswith("Feat_")]
    
    X_fit, X_test = train_df[feat_cols].values, test_df[feat_cols].values

    # Pre-processing
    X_fit_lat, X_fit_hc = split_lat_hc(X_fit)
    X_test_lat, X_test_hc = split_lat_hc(X_test)

    # Variance Thresholding
    vt = VarianceThreshold(threshold=1e-6)
    X_fit_full, X_test_full = vt.fit_transform(X_fit), vt.transform(X_test)
    X_fit_lat, X_test_lat = vt.fit_transform(X_fit_lat), vt.transform(X_test_lat)
    X_fit_hc, X_test_hc = vt.fit_transform(X_fit_hc), vt.transform(X_test_hc)

    # Scaling
    sc = StandardScaler()
    X_fit_full = sc.fit_transform(X_fit_full)
    X_test_full = sc.transform(X_test_full)
    
    # RFE
    rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=15)
    X_fit_rfe = rfe.fit_transform(X_fit_full, y_fit)
    X_test_rfe = rfe.transform(X_test_full)

    feature_sets = {
        "Handcrafted": (X_fit_hc, X_test_hc),
        "Latent": (X_fit_lat, X_test_lat),
        "Combined": (X_fit_full, X_test_full),
        "Combined_RFE": (X_fit_rfe, X_test_rfe)
    }

    results = []
    for clf_name, clf in get_classifiers().items():
        for feat_name, (X_tr, X_te) in feature_sets.items():
            clf.fit(X_tr, y_fit)
            y_pred = clf.predict(X_te)
            
            results.append({
                "Variant": mode_variant,
                "Classifier": clf_name,
                "Feature Set": feat_name,
                "Test Acc": round(accuracy_score(y_test, y_pred), 4),
                "Balanced Acc": round(balanced_accuracy_score(y_test, y_pred), 4),
                "Macro F1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
            })

    # Salvataggio risultati specifici
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_PATH, f"metrics_{mode_variant}.csv"), index=False)
    print(res_df.to_string(index=False))

# ═════════════════════════════════════════════
# 3. MAIN
# ═════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_mode", default="autoencoder", help="Arch base (autoencoder o classifier)")
    args = parser.parse_args()

    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Trova tutte le varianti (_original, _coral, _mmd)
    modes = discover_variants(args.arch_mode)
    
    if not modes:
        print(f"Nessuna variante trovata per {args.arch_mode}. Assicurati che i file siano presenti in {CHECKPOINT_PATH} con il prefisso {FEATURE_PREFIX}.")
    else:
        for m in modes:
            run_ablation(m)