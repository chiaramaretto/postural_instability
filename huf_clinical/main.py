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

def load_fusion_data_mmap(split):
    """Utility per caricare le feature dei 6 DR-SAE senza saturare la RAM."""
    feats = []
    for col in SENSOR_COLS:
        feat_path = os.path.join(FEATURE_DIR, f"{split}_{col}_feat.npy")
        # Carica in modalità memory-map: legge dal disco solo quando serve
        feats.append(np.load(feat_path, mmap_mode='r'))
    
    # Concatenazione avviene asse per asse per risparmiare memoria
    return np.concatenate(feats, axis=1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in [CHECKPOINT_DIR, FEATURE_DIR, EXTRACTED_DIR]:
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # STEP 1: DR-SAE (Axis-wise Representation)[cite: 7]
    # =========================================================================
    for col in SENSOR_COLS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"dr_sae_{col}.pth")
        train_axis_path = os.path.join(CHANNELS_PATH, "train", f"{col}.npy")
        
        # Caricamento mmap per calcolare le statistiche
        train_data_mmap = np.load(train_axis_path, mmap_mode='r')
        mean_tr, std_tr = train_data_mmap.mean(), train_data_mmap.std()
        
        model_dr = DR_SAE().to(device)
        if not os.path.exists(ckpt_path):
            print(f"\n--- Training DR-SAE: {col} ---")
            # Carica in RAM solo il necessario per il training
            train_norm = (np.array(train_data_mmap) - mean_tr) / (std_tr + 1e-8)
            train_tensor = torch.from_numpy(train_norm).float().unsqueeze(1)
            loader = DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True)
            
            model_dr = train_stacked_dr_sae(model_dr, loader, device, max_epochs=20)
            torch.save(model_dr.state_dict(), ckpt_path)
            
            # Pulizia immediata della memoria
            del train_norm, train_tensor, loader
            gc.collect()
            torch.cuda.empty_cache()
        else:
            model_dr.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Estrazione feature axis-wise con batching pesante per la RAM
        for split in ["train", "val", "test"]:
            split_axis_path = os.path.join(CHANNELS_PATH, split, f"{col}.npy")
            if not os.path.exists(split_axis_path): continue
            
            # Legge via mmap per l'estrazione
            split_data_mmap = np.load(split_axis_path, mmap_mode='r')
            model_dr.eval()
            out_feat_path = os.path.join(FEATURE_DIR, f"{split}_{col}_feat.npy")
            
            all_feats = []
            with torch.no_grad():
                # Processiamo a blocchi per non riempire la RAM
                for i in range(0, len(split_data_mmap), 128):
                    batch = np.array(split_data_mmap[i:i+128])
                    batch = (batch - mean_tr) / (std_tr + 1e-8)
                    batch_t = torch.from_numpy(batch).float().unsqueeze(1).to(device)
                    
                    _, feat = model_dr(batch_t)
                    all_feats.append(feat.cpu().numpy())
                    
                    del batch_t, feat
                
                # Salva su disco e libera subito la lista
                np.save(out_feat_path, np.concatenate(all_feats, axis=0))
                del all_feats
                gc.collect()
            
        del model_dr, train_data_mmap
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # STEP 2: LFF-AE (Local Feature Fusion)[cite: 7]
    # =========================================================================
    model_lff = LFF_AE(input_channels=6 * 256).to(device)
    ckpt_lff_base = os.path.join(CHECKPOINT_DIR, "lff_ae_base.pth")
    ckpt_lff_clinical = os.path.join(CHECKPOINT_DIR, "lff_ae_clinical.pth")

    # Fase 2A: Unsupervised
    if not os.path.exists(ckpt_lff_base):
        print("\n--- Training LFF-AE (Unsupervised) ---")
        X_train_fusion = load_fusion_data_mmap("train")
        train_tensor = torch.from_numpy(X_train_fusion).float()
        loader_fusion = DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True)
        
        model_lff = train_fusion_block(model_lff, loader_fusion, device, max_epochs=20, block_name="LFF")
        torch.save(model_lff.state_dict(), ckpt_lff_base)
        
        del X_train_fusion, train_tensor, loader_fusion
        gc.collect()
    else:
        model_lff.load_state_dict(torch.load(ckpt_lff_base, map_location=device))

    # Fase 2B: Clinical Fine-tuning[cite: 7]
    if not os.path.exists(ckpt_lff_clinical):
        print("\n--- Fine-tuning LFF-AE (Clinical) ---")
        X_train_fusion = load_fusion_data_mmap("train")
        y_train = np.load(os.path.join(BASE_DATA_PATH, "train", "labels.npy"))
        
        train_t = torch.from_numpy(X_train_fusion).float()
        target_t = torch.from_numpy(y_train).float()
        loader_lff = DataLoader(TensorDataset(train_t, target_t, torch.ones(len(y_train), dtype=torch.bool)), 
                                batch_size=16, shuffle=True)
        
        model_lff = fine_tune_lff_clinical(model_lff, loader_lff, device, max_epochs=15)
        torch.save(model_lff.state_dict(), ckpt_lff_clinical)
        
        del X_train_fusion, y_train, train_t, target_t, loader_lff
        gc.collect()
    else:
        model_lff.load_state_dict(torch.load(ckpt_lff_clinical, map_location=device))

    # =========================================================================
    # STEP 3: ESTRAZIONE FINALE E CSV[cite: 7]
    # =========================================================================
    for split in ["train", "val", "test"]:
        print(f"\nGenerazione CSV finale: {split}")
        X_fusion_mmap = load_fusion_data_mmap(split)
        metadata = pd.read_csv(os.path.join(BASE_DATA_PATH, split, "metadata.csv"))
        labels = np.load(os.path.join(BASE_DATA_PATH, split, "labels.npy"))

        model_lff.eval()
        final_feats_list = []
        with torch.no_grad():
            # Estrazione a mini-batch per non saturare la RAM nel pooling finale
            for i in range(0, len(X_fusion_mmap), 128):
                batch_t = torch.from_numpy(np.array(X_fusion_mmap[i:i+128])).float().to(device)
                _, final_latent = model_lff(batch_t)
                final_feats_list.append(torch.mean(final_latent, dim=2).cpu().numpy())
                del batch_t, final_latent
            
            final_feats = np.concatenate(final_feats_list, axis=0)

        df_final = pd.concat([metadata, pd.DataFrame(final_feats, columns=[f"feat_{i}" for i in range(final_feats.shape[1])])], axis=1)
        df_final['label'] = labels
        df_final.to_csv(os.path.join(EXTRACTED_DIR, f"huf_features_{split}.csv"), index=False)
        
        del X_fusion_mmap, metadata, labels, final_feats, final_feats_list
        gc.collect()

    print("\nProcesso completato con gestione ottimizzata della memoria.")

if __name__ == "__main__":
    main()