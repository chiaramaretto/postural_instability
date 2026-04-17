from train import train_model
import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

def main():
    # features
    features = pd.read_csv('posturalInstability/data/extracted_features/extracted_huf_features.csv')
    folder = 'posturalInstability/data/cleaned_data/'
    
    # clinical data
    clinical_data = pd.DataFrame()
    for file in os.listdir(folder):
        if file.endswith('clinical_data.csv'):
            df = pd.read_csv(os.path.join(folder, file))
            # Identifichiamo il dataset dal nome del file
            df['dataset'] = file.replace('_clinical_data.csv', '')
            clinical_data = pd.concat([clinical_data, df], ignore_index=True)
    
    merged_data = pd.merge(features, clinical_data, on=['subjectID', 'dataset'])
    metadata_cols = ['subjectID', 'dataset', 'postural_stability', 'berg', 'fes-i', 'updrs_iii', 'taskID', 'sessionID']
    train_df = merged_data[merged_data['dataset'] != 'fog_star'].copy()
    train_df = train_df.dropna(subset=['postural_stability'])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(train_df, groups=train_df['subjectID']))
    
    train_set = train_df.iloc[train_idx]
    test_set = train_df.iloc[test_idx]

    X_train = train_set.drop(columns=[c for c in metadata_cols if c in train_set.columns])
    y_train = train_set['postural_stability'].astype(int)
    
    X_test = test_set.drop(columns=[c for c in metadata_cols if c in test_set.columns])
    y_test = test_set['postural_stability'].astype(int)

    # Training
    model = train_model(X_train.values, y_train.values, num_classes=5)

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        
        print("\n--- Risultati Test Set (Interno) ---")
        print(classification_report(y_test, preds.numpy()))

if __name__ == "__main__":
    main()