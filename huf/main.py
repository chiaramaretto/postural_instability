import torch
import numpy as np
import pandas as pd
import os
import gc 
from torch.utils.data import DataLoader, TensorDataset
from model import DR_SAE, LFF_AE
from train import train_stacked_dr_sae, train_fusion_block

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_dir = "posturalInstability/huf/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Load data and metadata
    metadata = pd.read_csv("posturalInstability/data/windowed_data/metadata.csv")
    subjects = metadata['subjectID'].unique()
    np.random.shuffle(subjects)
    
    train_subjects = subjects[:int(0.7 * len(subjects))]
    train_indices = metadata[metadata['subjectID'].isin(train_subjects)].index.tolist()
    
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # --- STEP 1: DR-SAE Training & Feature Extraction to Disk ---
    for col in sensor_cols:
        ckpt_path = f"{ckpt_dir}/dr_sae_{col}.pth"
        feat_path = f"{ckpt_dir}/features_{col}.pt" 
        
        data = np.load(f"posturalInstability/data/windowed_data/axes/{col}.npy")
        full_data = torch.from_numpy(data).float()
        
        model_dr = DR_SAE().to(device)

        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint for DR-SAE: {col}")
            model_dr.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"\n--- Training DR-SAE: {col} ---")
            train_data = torch.from_numpy(data[train_indices]).float()
            train_loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True)
            model_dr = train_stacked_dr_sae(model_dr, train_loader, device)
            torch.save(model_dr.state_dict(), ckpt_path)
            del train_data
        
        if not os.path.exists(feat_path):
            print(f"Extracting features for {col}...")
            model_dr.eval()
            with torch.no_grad():
                temp_loader = DataLoader(TensorDataset(full_data), batch_size=512, shuffle=False)
                feats_list = []
                for b in temp_loader:
                    _, f = model_dr(b[0].to(device))
                    feats_list.append(f.cpu())
                all_axis_feats = torch.cat(feats_list, dim=0)
                torch.save(all_axis_feats, feat_path)
                del feats_list, all_axis_feats 
        
        del data, full_data, model_dr
        gc.collect()
        torch.cuda.empty_cache()

    # 2. STEP 2: Local Feature Fusion (LFF-AE) - FINAL FUSION
    ckpt_lff = f"{ckpt_dir}/lff_ae.pth"

    print("\nPreparing lazy dataset for LFF-AE (memory efficient)...")

    # --- Dataset custom ---
    class LFFDataset(torch.utils.data.Dataset):
        def __init__(self, feature_paths, indices=None):
            self.feature_paths = feature_paths
            self.indices = indices if indices is not None else None

            sample = torch.load(feature_paths[0], map_location='cpu')
            self.length = sample.shape[0]
            del sample

        def __len__(self):
            return len(self.indices) if self.indices is not None else self.length

        def __getitem__(self, idx):
            real_idx = self.indices[idx] if self.indices is not None else idx

            feats = []
            for p in self.feature_paths:
                f = torch.load(p, map_location='cpu')  
                feats.append(f[real_idx])              
                del f

            x = torch.cat(feats, dim=0) 
            return x


    feature_paths = [f"{ckpt_dir}/features_{col}.pt" for col in sensor_cols]

    train_dataset = LFFDataset(feature_paths, train_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model_lff = LFF_AE(input_channels=6*256).to(device)

    if os.path.exists(ckpt_lff):
        print("Loading checkpoint for LFF-AE")
        model_lff.load_state_dict(torch.load(ckpt_lff, map_location=device))
    else:
        print("\n--- Training Local Feature Fusion (LFF-AE) ---")
        
        model_lff = train_fusion_block(model_lff, train_loader, device, block_name="LFF")
        torch.save(model_lff.state_dict(), ckpt_lff)

        gc.collect()
        
    # --- FINAL STEP: Extract features from LFF-AE ---
    print("\n--- Final features extraction from LFF-AE ---")
    model_lff.eval()
    with torch.no_grad():
        lff_loader_full = DataLoader(TensorDataset(lff_input_all), batch_size=256, shuffle=False)
        final_list = []
        for b in lff_loader_full:
            _, lff_feat = model_lff(b[0].to(device))
            final_list.append(lff_feat.cpu())
        
        final_features = torch.cat(final_list, dim=0)
        
        del lff_input_all, final_list
        gc.collect()
        torch.cuda.empty_cache()
    final_features_flat = torch.mean(final_features, dim=2).numpy()
    
    # Concatenate metadata with the 256 extracted features
    df_results = pd.concat([metadata, pd.DataFrame(final_features_flat)], axis=1)
    df_results.to_csv("posturalInstability/data/extracted_huf_features.csv", index=False)
    
    print(f"Extraction completed. Feature vector size: {final_features_flat.shape[1]}")
    print("Results saved to: posturalInstability/data/extracted_huf_features.csv")

if __name__ == "__main__":
    main()