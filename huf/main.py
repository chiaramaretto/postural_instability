import torch
import numpy as np
import pandas as pd
import os
import gc 
from torch.utils.data import DataLoader, Dataset
from model import DR_SAE, LFF_AE
from train import train_stacked_dr_sae, train_fusion_block

# Memory-efficient Dataset using Memory Mapping
class MmapLFFDataset(Dataset):
    def __init__(self, feature_paths, indices=None):
        # We look for .npy files specifically for mmap
        self.feature_paths = [p.replace(".pt", ".npy") for p in feature_paths]
        self.indices = indices
        
        # Open memory maps - this maps file addresses without loading data into RAM
        self.data_views = [np.load(p, mmap_mode='r') for p in self.feature_paths]
        
        if self.indices is not None:
            self.length = len(self.indices)
        else:
            self.length = self.data_views[0].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        
        # Pull only the specific slice into memory and convert to tensor
        # Use .copy() to ensure the memory is ownable by PyTorch
        feats = [torch.from_numpy(view[real_idx].copy()).float() for view in self.data_views]
        
        # Concatenate 6 sensors along the channel dimension
        return torch.cat(feats, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_dir = "posturalInstability/huf/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Load metadata
    metadata = pd.read_csv("posturalInstability/data/windowed_data/metadata.csv")
    subjects = metadata['subjectID'].unique()
    np.random.shuffle(subjects)
    
    train_subjects = subjects[:int(0.7 * len(subjects))]
    train_indices = metadata[metadata['subjectID'].isin(train_subjects)].index.tolist()
    
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    feature_paths = [f"{ckpt_dir}/features_{col}.npy" for col in sensor_cols]

    # --- STEP 1: DR-SAE Training & Feature Extraction ---
    for col in sensor_cols:
        ckpt_path = f"{ckpt_dir}/dr_sae_{col}.pth"
        feat_path = f"{ckpt_dir}/features_{col}.npy" 
        
        if not os.path.exists(feat_path):
            # Load raw data only when needed
            raw_data = np.load(f"posturalInstability/data/windowed_data/axes/{col}.npy")
            model_dr = DR_SAE().to(device)

            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint for DR-SAE: {col}")
                model_dr.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                print(f"\n--- Training DR-SAE: {col} ---")
                train_data = torch.from_numpy(raw_data[train_indices]).float()
                train_loader = DataLoader(torch.utils.data.TensorDataset(train_data), batch_size=64, shuffle=True)
                model_dr = train_stacked_dr_sae(model_dr, train_loader, device)
                torch.save(model_dr.state_dict(), ckpt_path)
                del train_data

            print(f"Extracting features for {col} to disk...")
            model_dr.eval()
            all_feats = []
            full_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(raw_data).float()), batch_size=512)
            
            with torch.no_grad():
                for b in full_loader:
                    _, f = model_dr(b[0].to(device))
                    all_feats.append(f.cpu().numpy())
            
            # Save as NumPy for memory mapping later
            np.save(feat_path, np.concatenate(all_feats, axis=0))
            
            del raw_data, model_dr, all_feats
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"Features for {col} already exist. Skipping.")

    # --- STEP 2: Local Feature Fusion (LFF-AE) ---
    ckpt_lff = f"{ckpt_dir}/lff_ae.pth"
    train_dataset = MmapLFFDataset(feature_paths, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0 )

    

    model_lff = LFF_AE(input_channels=6*256).to(device)

    if os.path.exists(ckpt_lff):
        print("Loading checkpoint for LFF-AE")
        model_lff.load_state_dict(torch.load(ckpt_lff, map_location=device))
    else:
        print("\n--- Training Local Feature Fusion (LFF-AE) ---")
        model_lff = train_fusion_block(model_lff, train_loader, device, block_name="LFF")
        torch.save(model_lff.state_dict(), ckpt_lff)

    # --- FINAL STEP: Extract features from LFF-AE ---
    print("\n--- Final features extraction from LFF-AE ---")
    model_lff.eval()
    
    # Use full dataset (no indices) for extraction
    full_lff_dataset = MmapLFFDataset(feature_paths)
    extract_loader = DataLoader(full_lff_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    final_list = []
    with torch.no_grad():
        for b in extract_loader:
            _, lff_feat = model_lff(b.to(device))
            # Temporal Global Average Pooling as per standard paper practices
            final_list.append(torch.mean(lff_feat, dim=2).cpu())
    
    final_features_flat = torch.cat(final_list, dim=0).numpy()
    
    # 3. Save Results
    df_results = pd.concat([metadata, pd.DataFrame(final_features_flat)], axis=1)
    df_results.to_csv("posturalInstability/data/extracted_huf_features.csv", index=False)
    
    print(f"Extraction completed. Feature vector size: {final_features_flat.shape[1]}")
    print("Results saved to: posturalInstability/data/extracted_huf_features.csv")

if __name__ == "__main__":
    main()