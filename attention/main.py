import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit

from model import SubjectTaskAttentionModel
from train import fit_model, evaluate

class SubjectBagDataset(Dataset):
    """Dataset che organizza le finestre in 'borse' per ogni soggetto."""
    def __init__(self, bags):
        self.bags = bags

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx]

def collate_subject_bags(batch):
    """Gestisce il padding dinamico per soggetti con numero diverso di finestre."""
    max_len = max(item["x"].shape[0] for item in batch)
    feat_dim = batch[0]["x"].shape[1]

    x = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    task_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    window_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    y = torch.zeros((len(batch),), dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        x_np = np.array(item["x"], copy=True)
        task_np = np.array(item["task_ids"], copy=True)
        x[i, :n] = torch.from_numpy(x_np).float()
        task_ids[i, :n] = torch.from_numpy(task_np).long()
        window_mask[i, :n] = True
        y[i] = item["y"]

    return {
        "x": x,
        "task_ids": task_ids,
        "window_mask": window_mask,
        "y": y
    }

def main():
    print("\n" + "="*50)
    print("ATTENTION REGRESSOR (POSTURAL STABILITY)")
    print("="*50)

    # 1. Setup percorsi (Assumi link simbolico a Drive già creato)
    base_path = "posturalInstability/data"
    features_path = os.path.join(base_path, "extracted_features", "features.csv")

    print(f"[1/5] Caricamento dati...")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"File non trovato: {features_path}. Verifica il link al Drive.")

    # Carica feature e metadati clinici
    features_df = pd.read_csv(features_path, dtype={'subjectID': str, 'dataset': str}, low_memory=False)

    # Normalizza le chiavi usate per merge/split
    features_df['subjectID'] = features_df['subjectID'].astype(str)
    features_df['dataset'] = features_df['dataset'].astype(str)

    # Identifica le colonne delle feature HUF (colonne numeriche 0, 1, 2...)
    feature_cols = [c for c in features_df.columns if c.isdigit()]
    
    print(f"      Dati caricati: {len(features_df)} finestre per {features_df[['subjectID', 'dataset']].drop_duplicates().shape[0]} soggetti-dataset.")

    # 2. Creazione dei "Subject Bags"
    print(f"[2/5] Raggruppamento per soggetto e dataset...")
    subject_bags = []
    # Usiamo postural_stability originale (0-4) come target continuo.
    for (subj_id, dataset), group in features_df.groupby(['subjectID', 'dataset']):
        # Controlla se il target (postural_stability) è disponibile
        y_val = pd.to_numeric(group['postural_stability'].iloc[0], errors='coerce')
        if pd.isna(y_val):
            continue  # Salta soggetti senza target
        
        if y_val < 0 or y_val > 4:
            continue  # Salta soggetti con target fuori range 0-4
        
        subject_bags.append({
            "subjectID": subj_id,
            "dataset": dataset,
            "bag_id": f"{subj_id}__{dataset}",
            "x": group[feature_cols].values,
            "task_ids": group['taskID'].values,
            "y": float(y_val),
        })

    # 3. Split Allenamento/Validazione (per soggetto)
    bag_groups = np.array([bag["bag_id"] for bag in subject_bags])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(subject_bags, groups=bag_groups))

    train_bags = [subject_bags[i] for i in train_idx]
    val_bags = [subject_bags[i] for i in val_idx]

    train_ds = SubjectBagDataset(train_bags)
    val_ds = SubjectBagDataset(val_bags)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_subject_bags)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_subject_bags)

    # 4. Training
    print(f"[3/5] Inizializzazione modello Hierarchical Attention...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SubjectTaskAttentionModel(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_tasks=3, # Task 0, 1, 2
        dropout=0.3
    ).to(device)

    print(f"[4/5] Inizio training (MSE Loss)...")
    model = fit_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        max_epochs=80,
        patience=12,
        min_delta=1e-3,
        scheduler_patience=5,
        scheduler_factor=0.5,
        min_lr=1e-5,
    )

    # 5. Valutazione Finale
    print(f"\n[5/5] Valutazione finale su Test Set (Soggetti mai visti):")
    metrics = evaluate(model, val_loader, device)
    print("\nMetriche di regressione:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Rounded Accuracy (0-4): {metrics['acc_rounded']:.4f}")

    print("\n" + "="*50)
    print("✓ CLASSIFICATORE COMPLETATO")
    print("="*50)

if __name__ == "__main__":
    main()