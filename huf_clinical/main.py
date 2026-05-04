import os
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from model import DR_SAE, LFF_AE
from train import train_stacked_dr_sae, train_fusion_block, fine_tune_lff_clinical

# --- CONFIGURAZIONE PERCORSI ---
BASE_DATA_PATH = "posturalInstability/huf_clinical/data/windowed_data"
CHANNELS_PATH = os.path.join(BASE_DATA_PATH, "channels")
EXTRACTED_DIR = "posturalInstability/huf_clinical/data/extracted_features"
CHECKPOINT_DIR = "posturalInstability/huf_clinical/data/checkpoints"
FEATURE_DIR = "posturalInstability/huf_clinical/data/axis_features"

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

def load_fusion_data(split):
    """Utility per concatenare le feature estratte dai 6 DR-SAE."""
    feats = []
    for col in SENSOR_COLS:
        feat_path = os.path.join(FEATURE_DIR, f"{split}_{col}_feat.npy")
        feats.append(np.load(feat_path))
    return np.concatenate(feats, axis=1) # Shape: [N, 1536, L] dove 1536 = 6 assi * 256 canali

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in [CHECKPOINT_DIR, FEATURE_DIR, EXTRACTED_DIR]:
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # STEP 1: DR-SAE (Axis-wise Representation)
    # =========================================================================
    for col in SENSOR_COLS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"dr_sae_{col}.pth")
        
        # Caricamento dati di train per normalizzazione e training
        train_axis_path = os.path.join(CHANNELS_PATH, "train", f"{col}.npy")
        train_data = np.load(train_axis_path).astype(np.float32)
        
        # Statistiche fisse basate solo sul TRAIN per evitare leakage[cite: 7, 8]
        mean_tr, std_tr = train_data.mean(), train_data.std()
        
        model_dr = DR_SAE().to(device)
        if not os.path.exists(ckpt_path):
            print(f"\n--- Training DR-SAE: {col} ---")
            train_norm = (train_data - mean_tr) / (std_tr + 1e-8)
            train_tensor = torch.from_numpy(train_norm).unsqueeze(1)
            loader = DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True)
            
            model_dr = train_stacked_dr_sae(model_dr, loader, device, max_epochs=20)
            torch.save(model_dr.state_dict(), ckpt_path)
            del train_norm, train_tensor
        else:
            model_dr.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Estrazione feature per tutti gli split usando le statistiche del train
        for split in ["train", "val", "test"]:
            split_axis_path = os.path.join(CHANNELS_PATH, split, f"{col}.npy")
            if not os.path.exists(split_axis_path): continue
            
            split_data = np.load(split_axis_path).astype(np.float32)
            split_data = (split_data - mean_tr) / (std_tr + 1e-8)
            
            model_dr.eval()
            out_feat_path = os.path.join(FEATURE_DIR, f"{split}_{col}_feat.npy")
            
            with torch.no_grad():
                loader_ext = DataLoader(TensorDataset(torch.from_numpy(split_data).unsqueeze(1)), batch_size=128)
                all_feats = []
                for batch in loader_ext:
                    _, feat = model_dr(batch[0].to(device))
                    all_feats.append(feat.cpu().numpy())
                np.save(out_feat_path, np.concatenate(all_feats, axis=0))
            
        del model_dr, train_data; gc.collect()

    # =========================================================================
    # STEP 2: LFF-AE (Local Feature Fusion)[cite: 7]
    # =========================================================================
    ckpt_lff_base = os.path.join(CHECKPOINT_DIR, "lff_ae_base.pth")
    ckpt_lff_clinical = os.path.join(CHECKPOINT_DIR, "lff_ae_clinical.pth")
    model_lff = LFF_AE(input_channels=6 * 256).to(device)

    X_train_fusion = load_fusion_data("train")
    y_train = np.load(os.path.join(BASE_DATA_PATH, "train", "labels.npy"))

    # FASE 2A: Addestramento Non Supervisionato (Ricostruzione)[cite: 7]
    if not os.path.exists(ckpt_lff_base):
        print("\n--- Training LFF-AE (Unsupervised Reconstruction) ---")
        train_tensor = torch.from_numpy(X_train_fusion)
        loader_fusion = DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True)
        model_lff = train_fusion_block(model_lff, loader_fusion, device, max_epochs=20, block_name="LFF")
        torch.save(model_lff.state_dict(), ckpt_lff_base)
        del train_tensor
    else:
        model_lff.load_state_dict(torch.load(ckpt_lff_base, map_location=device))

    # FASE 2B: Fine-tuning Supervisionato (Clinical-aware)[cite: 7]
    if not os.path.exists(ckpt_lff_clinical):
        print("\n--- Fine-tuning LFF-AE (Clinical-aware) ---")
        train_t = torch.from_numpy(X_train_fusion)
        target_t = torch.from_numpy(y_train.astype(np.float32))
        # Maschera 'True' per tutti i dati di train poiché etichettati
        loader_lff = DataLoader(TensorDataset(train_t, target_t, torch.ones(len(y_train), dtype=torch.bool)), 
                                batch_size=16, shuffle=True)
        
        model_lff = fine_tune_lff_clinical(model_lff, loader_lff, device, max_epochs=15)
        torch.save(model_lff.state_dict(), ckpt_lff_clinical)
        del train_t, target_t
    else:
        model_lff.load_state_dict(torch.load(ckpt_lff_clinical, map_location=device))

    # =========================================================================
    # STEP 3: ESTRAZIONE FINALE E GENERAZIONE CSV[cite: 7]
    # =========================================================================
    for split in ["train", "val", "test"]:
        print(f"\nGenerazione CSV finale per lo split: {split}")
        X_fusion = load_fusion_data(split)
        metadata = pd.read_csv(os.path.join(BASE_DATA_PATH, split, "metadata.csv"))
        labels = np.load(os.path.join(BASE_DATA_PATH, split, "labels.npy"))

        model_lff.eval()
        with torch.no_grad():
            # Il pooling temporale (mean) riduce le feature a un vettore flat per finestra[cite: 7]
            _, final_latent = model_lff(torch.from_numpy(X_fusion).to(device))
            final_feats = torch.mean(final_latent, dim=2).cpu().numpy()

        # Creazione DataFrame con feature estratte[cite: 7]
        df_feats = pd.DataFrame(final_feats, columns=[f"feat_{i}" for i in range(final_feats.shape[1])])
        df_final = pd.concat([metadata, df_feats], axis=1)
        df_final['label'] = labels

        out_csv = os.path.join(EXTRACTED_DIR, f"huf_features_{split}.csv")
        df_final.to_csv(out_csv, index=False)
        print(f"File salvato correttamente: {out_csv}")

    print("\nProcesso completato. Le feature HUF sono pronte per l'analisi statistica.")

if __name__ == "__main__":
    main()