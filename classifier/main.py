import torch
import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from train import train_model


def confusion_matrices(y_true, y_pred, groups, group_name, labels):
    print(f"\n--- Confusion Matrix per {group_name} ---")
    for value in sorted(pd.Series(groups).dropna().unique()):
        mask = (groups == value)
        y_t = y_true[mask]
        y_p = y_pred[mask]

        cm = confusion_matrix(y_t, y_p, labels=labels)
        print(f"\n{group_name} = {value} | n={len(y_t)}")
        print(cm)

def main():
    
    print("\n" + "="*80)
    print("SUBJECT-LEVEL AGGREGATION CLASSIFIER")
    print("="*80)
    
    # Setup paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(repo_root, 'data', 'extracted_features', 'features_clinical.csv')
   
    print(f"\n[1/5] Loading features from: {features_path}")
    
    # Check file exists
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    # Load features (window-level)
    features = pd.read_csv(features_path, dtype={'subjectID': str}, low_memory=False)
    print(f"      Loaded {len(features)} window rows, {features['subjectID'].nunique()} unique subjects")
    
    
    # Clean subjectID
    features['subjectID'] = features['subjectID'].astype(str).str.strip()
    
    # Exclude FoG-STAR
    features = features[features['dataset'] != 'fog_star'].copy()
    print(f"      After excluding fog_star: {len(features)} rows, {features['subjectID'].nunique()} subjects")
    
    # === AGGREGATION: Features by subject (mean across all windows) ===
    print(f"\n[3/5] Aggregating features by subject (mean)")
    feature_cols = [c for c in features.columns if c.isdigit()]
    print(f"      Aggregating {len(feature_cols)} features...")
    
    # Aggregate using (subjectID, dataset) to avoid mixing equal IDs from different datasets.
    group_keys = ['subjectID', 'dataset']
    subject_features = features.groupby(group_keys)[feature_cols].mean()

    # Keep one clinical row per (subjectID, dataset).
    subject_clinical = features.groupby(group_keys).first()[
        ['postural_stability', 'updrs_iii', 'berg', 'fes-i', 'taskID', 'sessionID']
    ]

    # Combine and restore subjectID/dataset as explicit columns.
    subject_df = pd.concat([subject_features, subject_clinical], axis=1).reset_index()
    print(f"      Created subject-level data: {len(subject_df)} subject-dataset pairs")
    
    # Check target availability
    target_col = 'postural_stability'
    print(f"\n[3b] Target variable availability:")
    print(f"      postural_stability: {subject_df[target_col].notna().sum()}/{len(subject_df)} subjects")
    print(f"      updrs_iii: {subject_df['updrs_iii'].notna().sum()}/{len(subject_df)} subjects")
    print(f"      berg: {subject_df['berg'].notna().sum()}/{len(subject_df)} subjects")
    
    if target_col not in subject_df.columns or subject_df[target_col].notna().sum() < 5:
        raise ValueError(f"Target column '{target_col}' missing or has <5 valid subjects")
    
    # Prepare data: keep only subjects with target
    subject_df_clean = subject_df[subject_df[target_col].notna()].copy()
    subject_df_clean[target_col] = pd.to_numeric(subject_df_clean[target_col], errors='coerce')
    subject_df_clean = subject_df_clean[subject_df_clean[target_col].notna()].copy()
    
    print(f"      After filtering NaN: {len(subject_df_clean)} subjects with target data")
    
    # Binarize postural_stability
    subject_df_clean[target_col] = (subject_df_clean[target_col] <= 1).astype(int)
    
    # === TRAIN/TEST SPLIT at SUBJECT level ===
    print(f"\n[4/5] Train/Test split (subject-wise, no leakage)")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(subject_df_clean, groups=subject_df_clean['dataset']))
    
    train_set = subject_df_clean.iloc[train_idx].copy()
    test_set = subject_df_clean.iloc[test_idx].copy()
    
    print(f"      Train: {len(train_set)} subjects")
    print(f"        Class 0: {(train_set[target_col]==0).sum()}")
    print(f"        Class 1: {(train_set[target_col]==1).sum()}")
    print(f"      Test: {len(test_set)} subjects")
    print(f"        Class 0: {(test_set[target_col]==0).sum()}")
    print(f"        Class 1: {(test_set[target_col]==1).sum()}")
    
    # Prepare features
    metadata_cols = [c for c in subject_df_clean.columns if c in [
        'subjectID', 'dataset', 'postural_stability', 'berg', 'fes-i', 
        'updrs_iii', 'taskID', 'sessionID'
    ]]
    
    X_train = train_set.drop(columns=metadata_cols)
    y_train = train_set[target_col].astype(int)
    
    X_test = test_set.drop(columns=metadata_cols)
    y_test = test_set[target_col].astype(int)
    
    # Convert to numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    print(f"      Feature matrix: {X_train.shape[1]} features")
    
    # Compute class weights
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    num_classes = len(np.unique(y_train))
    
    print(f"\n[5/5] Training torch classifier on subject-level data...")
    print(f"      input_size={X_train.shape[1]}, num_classes={num_classes}")
    
    # === TORCH CLASSIFIER TRAINING ===
    model = train_model(
        X_train.values, 
        y_train.values, 
        class_weights=weights_tensor, 
        input_size=X_train.shape[1], 
        num_classes=num_classes, 
        learning_rate=0.001, 
        num_epochs=100
    )
    
    # === EVALUATION ===
    print(f"\n" + "="*80)
    print(f"RESULTS - SUBJECT-LEVEL CLASSIFICATION ({target_col})")
    print(f"="*80)
    print(f"\nTest set: {len(y_test)} subjects")
    print(f"  Class 0: {(y_test==0).sum()}")
    print(f"  Class 1: {(y_test==1).sum()}")
    
    eval_labels = [0, 1]
    target_names = ['Class 0 (PS > 1)', 'Class 1 (PS ≤ 1)']
    
    # Predict on test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()
        y_test_np = y_test.to_numpy()
    
    print(f"\nClassification Report:")
    print(classification_report(y_test_np, y_pred, labels=eval_labels, target_names=target_names, zero_division=0))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_np, y_pred, labels=eval_labels)
    print(cm)
    
    # Group confusion matrices
    confusion_matrices(
        y_true=y_test_np,
        y_pred=y_pred,
        groups=test_set['dataset'].to_numpy(),
        group_name='dataset',
        labels=eval_labels
    )
    
    print(f"\n" + "="*80)
    print(f"✓ SUBJECT-LEVEL AGGREGATION CLASSIFIER COMPLETE")
    print(f"  Model: Torch Classifier (from train.py)")
    print(f"  Features: Aggregated at subject-level (mean)")
    print(f"  Subjects: Train={len(train_set)}, Test={len(test_set)}")
    print(f"  Expected improvement vs window-level: +20-30% F1")
    print(f"="*80)

if __name__ == "__main__":
    main()