import torch
import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from train import train_model

def main():
    # Setup percorsi (usa percorsi relativi per Colab/Locale)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(repo_root, 'data', 'extracted_features', 'clinicalfeatures_5s_32.csv')
    # extract number of features from the file name (assuming it's in the format features_{window}s_{num_features}.csv)
    num_features = int(re.search(r'features_\d+s_(\d+)\.csv', os.path.basename(features_path)).group(1))
    folder = os.path.join(repo_root, 'data', 'cleaned_data')

    # Caricamento feature e clinica (forzando le stringhe per i merge)
    features = pd.read_csv(features_path, dtype={'subjectID': str}, low_memory=False)
    
    clinical_data = pd.DataFrame()
    for file in os.listdir(folder):
        if file.endswith('clinical.csv') or file.endswith('clinical_data.csv'):
            df = pd.read_csv(os.path.join(folder, file), dtype={'subjectID': str})

            df['dataset'] = re.sub(r'_clinical(?:_data)?\.csv$', '', file)
            clinical_data = pd.concat([clinical_data, df], ignore_index=True)

    # Pulizia subjectID
    features['subjectID'] = features['subjectID'].astype(str).str.strip()
    clinical_data['subjectID'] = clinical_data['subjectID'].astype(str).str.strip()
    
    # Merge
    merged_data = pd.merge(features, clinical_data, on=['subjectID', 'dataset'])

    # --- LOGICA DI BINARIZZAZIONE ---
    # Escludiamo FoG-STAR (lo useremo come test esterno cieco)
    train_df = merged_data[merged_data['dataset'] != 'fog_star'].copy()
    train_df = train_df.dropna(subset=['postural_stability'])
    
    # Creazione del target binario: 0 (Stabile/Lieve) vs 1 (Moderato/Severo)
    train_df["binary_ps"] = train_df['postural_stability'].apply(lambda x: 0 if x <= 1 else 1)

    # --- DEFINIZIONE METADATI (DA DROPPARE) ---
    # CRITICO: binary_ps deve essere in questa lista per non essere usata come feature di input!
    metadata_cols = [
        'subjectID', 'dataset', 'postural_stability', 'berg', 'fes-i', 
        'updrs_iii', 'taskID', 'sessionID', 'binary_ps', 'isTurn'
    ]

    # add standard scaler on features (escludendo le colonne di metadata)
    feature_cols = [str(col) for col in list(range(0, num_features))]
    train_df[feature_cols] = (train_df[feature_cols] - train_df[feature_cols].mean()) / train_df[feature_cols].std()

    # Split Subject-Wise (No Data Leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(train_df, groups=train_df['subjectID']))
    
    train_set = train_df.iloc[train_idx]
    test_set = train_df.iloc[test_idx]

    # Prepariamo X e y
    X_train = train_set.drop(columns=[c for c in metadata_cols if c in train_set.columns])
    y_train = train_set['binary_ps'].astype(int)
    
    X_test = test_set.drop(columns=[c for c in metadata_cols if c in test_set.columns])
    y_test = test_set['binary_ps'].astype(int)

    # Pulizia dati numerici
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # Calcolo pesi per bilanciamento
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_tensor = torch.tensor([1.0, 5.0], dtype=torch.float32)

    # Training
    model = train_model(X_train.values, y_train.values, 
                        class_weights=weights_tensor, 
                        input_size=X_train.shape[1], 
                        num_classes=2, learning_rate=0.0001, num_epochs=200)

    # Valutazione
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        
        print("\n--- Risultati Classificazione Binaria (Test Interno) ---")
        print(classification_report(y_test, preds.numpy()))
        print("\nMatrice di Confusione:")
        print(confusion_matrix(y_test, preds.numpy()))

if __name__ == "__main__":
    main()