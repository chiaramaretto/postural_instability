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
STAC_FEATURE_CACHE_DIR = "posturalInstability/huf_clinical/data/stac_feature_cache"

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
BATCH_SIZE = 64  # Aumentato per ridurre overhead
CHUNK_SIZE = 256  # Per processing in blocchi
FORCE_EXTRACT = False  # Imposta True per forzare re-estrazione features anche se in cache


def is_valid_npy_cache(file_path):
    """Verifica che un file .npy esista ed sia leggibile come mmap."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False

    try:
        np.load(file_path, mmap_mode='r')
        return True
    except Exception:
        return False


def save_npy_atomic(file_path, array):
    """Salva un .npy in modo atomico per evitare file parziali/corrotti."""
    temp_path = f"{file_path}.tmp"
    with open(temp_path, "wb") as handle:
        np.save(handle, array)
    os.replace(temp_path, file_path)


class AxisDataset:
    def __init__(self, axis_array, mean_value, std_value):
        self.axis_array = axis_array
        self.mean_value = mean_value
        self.std_value = std_value

    def __len__(self):
        return len(self.axis_array)

    def __getitem__(self, idx):
        sample = np.asarray(self.axis_array[idx], dtype=np.float32)
        sample = (sample - self.mean_value) / (self.std_value + 1e-8)
        return torch.from_numpy(sample).unsqueeze(0)


class FusionDataset:
    def __init__(self, feat_mmaps):
        self.feat_mmaps = feat_mmaps
        self.n_samples = len(feat_mmaps[0])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        batch_feats = np.concatenate([m[idx:idx + 1] for m in self.feat_mmaps], axis=1)
        return torch.from_numpy(batch_feats.astype(np.float32))


class FusionLabelDataset:
    def __init__(self, feat_mmaps, labels):
        self.feat_mmaps = feat_mmaps
        self.labels = labels
        self.n_samples = len(feat_mmaps[0])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        batch_feats = np.concatenate([m[idx:idx + 1] for m in self.feat_mmaps], axis=1)
        return (
            torch.from_numpy(batch_feats.astype(np.float32)),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

def load_fusion_data_mmap(split):
    """Carica le feature DR-SAE già estratte via mmap (non i dati grezzi)."""
    feats = []
    for col in SENSOR_COLS:
        feat_path = os.path.join(STAC_FEATURE_CACHE_DIR, f"{split}_{col}_feat.npy")
        if not is_valid_npy_cache(feat_path):
            raise FileNotFoundError(f"Cache features non valida o mancante: {feat_path}")
        feats.append(np.load(feat_path, mmap_mode='r'))
    return feats

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in [CHECKPOINT_DIR, STAC_FEATURE_CACHE_DIR, EXTRACTED_DIR]:
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # STEP 1: DR-SAE (Axis-wise Representation)
    # =========================================================================
    for col in SENSOR_COLS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"dr_sae_{col}.pth")
        train_axis_path = os.path.join(CHANNELS_PATH, "train", f"{col}.npy")
        
        # Carica solo statistiche via mmap
        train_data_mmap = np.load(train_axis_path, mmap_mode='r')
        mean_tr = float(train_data_mmap.mean())
        std_tr = float(train_data_mmap.std())
        
        model_dr = DR_SAE().to(device)
        if not os.path.exists(ckpt_path):
            print(f"\n--- Training DR-SAE: {col} ---")
            # Carica training normalizato
            train_data = np.load(train_axis_path, mmap_mode='c')
            train_tensor = torch.from_numpy(((train_data.astype(np.float32) - mean_tr) / (std_tr + 1e-8))).unsqueeze(1)
            loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

            val_axis_path = os.path.join(CHANNELS_PATH, "val", f"{col}.npy")
            val_loader = None
            if os.path.exists(val_axis_path):
                val_data = np.load(val_axis_path, mmap_mode='r')
                val_loader = DataLoader(
                    AxisDataset(val_data, mean_tr, std_tr),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )
            
            model_dr = train_stacked_dr_sae(model_dr, loader, device, val_loader=val_loader, max_epochs=20)
            torch.save(model_dr.state_dict(), ckpt_path)
            
            del train_data, train_tensor, loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
        else:
            model_dr.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Estrazione feature per split con chunking pesante
        for split in ["train", "val", "test"]:
            split_axis_path = os.path.join(CHANNELS_PATH, split, f"{col}.npy")
            if not os.path.exists(split_axis_path):
                continue
            
            out_feat_path = os.path.join(STAC_FEATURE_CACHE_DIR, f"{split}_{col}_feat.npy")
            
            # Cache: salta estrazione se già presente (a meno che FORCE_EXTRACT=True)
            if is_valid_npy_cache(out_feat_path) and not FORCE_EXTRACT:
                print(f"  ✓ Features già estratte: {split}_{col}")
                continue
            
            split_data_mmap = np.load(split_axis_path, mmap_mode='r')
            n_samples = len(split_data_mmap)
            
            print(f"  → Estrazione features: {split}_{col}")
            model_dr.eval()
            all_feats = []
            
            with torch.no_grad():
                for i in range(0, n_samples, CHUNK_SIZE):
                    chunk = split_data_mmap[i:i+CHUNK_SIZE].astype(np.float32)
                    chunk = (chunk - mean_tr) / (std_tr + 1e-8)
                    chunk_t = torch.from_numpy(chunk).unsqueeze(1).to(device)
                    
                    _, feat = model_dr(chunk_t)
                    all_feats.append(feat.cpu().numpy())
                    del chunk_t, feat
            
            # Salva e libera
            save_npy_atomic(out_feat_path, np.concatenate(all_feats, axis=0))
            del all_feats, split_data_mmap
            gc.collect()
        
        del model_dr, train_data_mmap
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # STEP 2: LFF-AE (Local Feature Fusion)
    # =========================================================================
    
    def get_fusion_batch(feat_mmaps, start_idx, end_idx):
        """Assembla un batch da feature mmap senza copie complete."""
        batch = np.concatenate([m[start_idx:end_idx] for m in feat_mmaps], axis=1)
        return batch.astype(np.float32)
    
    model_lff = LFF_AE(input_channels=6 * 256).to(device)
    ckpt_lff_base = os.path.join(CHECKPOINT_DIR, "lff_ae_base.pth")
    ckpt_lff_clinical = os.path.join(CHECKPOINT_DIR, "lff_ae_clinical.pth")

    # Fase 2A: Unsupervised
    if not os.path.exists(ckpt_lff_base):
        print("\n--- Training LFF-AE (Unsupervised) ---")
        feat_mmaps = load_fusion_data_mmap("train")
        dataset_fusion = FusionDataset(feat_mmaps)
        loader_fusion = DataLoader(dataset_fusion, batch_size=BATCH_SIZE, shuffle=True)

        val_feat_mmaps = load_fusion_data_mmap("val")
        val_dataset_fusion = FusionDataset(val_feat_mmaps)
        val_loader_fusion = DataLoader(val_dataset_fusion, batch_size=BATCH_SIZE, shuffle=False)
        
        model_lff = train_fusion_block(
            model_lff,
            loader_fusion,
            device,
            val_loader=val_loader_fusion,
            max_epochs=20,
            block_name="LFF",
        )
        torch.save(model_lff.state_dict(), ckpt_lff_base)
        
        del feat_mmaps, dataset_fusion, loader_fusion, val_feat_mmaps, val_dataset_fusion, val_loader_fusion
        gc.collect()
    else:
        model_lff.load_state_dict(torch.load(ckpt_lff_base, map_location=device))

    # Fase 2B: Clinical Fine-tuning
    if not os.path.exists(ckpt_lff_clinical):
        print("\n--- Fine-tuning LFF-AE (Clinical) ---")
        feat_mmaps = load_fusion_data_mmap("train")
        y_train = np.load(os.path.join(BASE_DATA_PATH, "train", "labels.npy"), mmap_mode='c').astype(np.float32)
        
        dataset_lff = FusionLabelDataset(feat_mmaps, y_train)
        loader_lff = DataLoader(dataset_lff, batch_size=BATCH_SIZE, shuffle=True)

        val_feat_mmaps = load_fusion_data_mmap("val")
        val_labels = np.load(os.path.join(BASE_DATA_PATH, "val", "labels.npy"), mmap_mode='c').astype(np.float32)
        val_dataset_lff = FusionLabelDataset(val_feat_mmaps, val_labels)
        val_loader_lff = DataLoader(val_dataset_lff, batch_size=BATCH_SIZE, shuffle=False)
        
        model_lff = fine_tune_lff_clinical(
            model_lff,
            loader_lff,
            device,
            val_loader=val_loader_lff,
            max_epochs=15,
        )
        torch.save(model_lff.state_dict(), ckpt_lff_clinical)
        
        del feat_mmaps, y_train, dataset_lff, loader_lff, val_feat_mmaps, val_labels, val_dataset_lff, val_loader_lff
        gc.collect()
    else:
        model_lff.load_state_dict(torch.load(ckpt_lff_clinical, map_location=device))

    # =========================================================================
    # STEP 3: ESTRAZIONE FINALE E CSV
    # =========================================================================
    for split in ["train", "val", "test"]:
        print(f"\nGenerazione CSV finale: {split}")
        feat_mmaps = load_fusion_data_mmap(split)
        metadata = pd.read_csv(os.path.join(BASE_DATA_PATH, split, "metadata.csv"))
        labels = np.load(os.path.join(BASE_DATA_PATH, split, "labels.npy"))
        n_samples = len(feat_mmaps[0])
        
        # Verifica che le features siano disponibili
        all_feats_exist = all(os.path.exists(os.path.join(STAC_FEATURE_CACHE_DIR, f"{split}_{col}_feat.npy")) 
                              for col in SENSOR_COLS)
        if all_feats_exist:
            print(f"  ✓ Riutilizzo features dal cache")
        
        model_lff.eval()
        final_feats_list = []
        
        with torch.no_grad():
            for i in range(0, n_samples, CHUNK_SIZE):
                batch = get_fusion_batch(feat_mmaps, i, min(i+CHUNK_SIZE, n_samples))
                batch_t = torch.from_numpy(batch).to(device)
                _, final_latent = model_lff(batch_t)
                final_feats_list.append(torch.mean(final_latent, dim=2).cpu().numpy())
                del batch_t, final_latent
            
            final_feats = np.concatenate(final_feats_list, axis=0)

        df_final = pd.concat([
            metadata, 
            pd.DataFrame(final_feats, columns=[f"feat_{i}" for i in range(final_feats.shape[1])])
        ], axis=1)
        df_final['label'] = labels
        df_final.to_csv(os.path.join(EXTRACTED_DIR, f"huf_features_{split}.csv"), index=False)
        
        del feat_mmaps, final_feats_list, final_feats
        gc.collect()

    print("\nProcesso completato con gestione ottimizzata della memoria.")

if __name__ == "__main__":
    main()