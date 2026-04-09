import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from model import DR_SAE, LFF_AE, GFF_AE
from train import train_stacked_dr_sae, train_fusion_block

def main():
    # Setup device and checkpoint directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_dir = "posturalInstability/huf/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Load data and metadata
    metadata = pd.read_csv("posturalInstability/data/windowed_data/metadata.csv")
    subjects = metadata['subjectID'].unique()
    np.random.shuffle(subjects)
    
    # Subject-based hold-out strategy (70% train, 30% test) [cite: 224]
    train_subjects = subjects[:int(0.7 * len(subjects))]
    train_indices = metadata[metadata['subjectID'].isin(train_subjects)].index.tolist()
    
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    axis_features = []

    # 2. STEP 1: Data Representation (DR-SAE)
    # Training independent AEs for each accelerometer and gyroscope signal [cite: 39]
    for col in sensor_cols:
        ckpt_path = f"{ckpt_dir}/dr_sae_{col}.pth"
        model_dr = DR_SAE().to(device)
        
        data = np.load(f"posturalInstability/data/windowed_data/axes/{col}.npy")
        full_data = torch.from_numpy(data).float()

        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint for DR-SAE: {col}")
            model_dr.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"\n--- Training DR-SAE: {col} ---")
            train_data = torch.from_numpy(data[train_indices]).float()
            train_loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True)
            
            # Layer-by-layer stacked training [cite: 102]
            model_dr = train_stacked_dr_sae(model_dr, train_loader, device)
            torch.save(model_dr.state_dict(), ckpt_path)
        
        model_dr.eval()
        with torch.no_grad():
            # Creiamo un loader temporaneo per non saturare la VRAM
            temp_loader = DataLoader(TensorDataset(full_data), batch_size=256, shuffle=False)
            feats = []
            for b in temp_loader:
                out, f = model_dr(b[0].to(device))
                feats.append(f.cpu()) # Spostiamo subito su CPU per liberare la GPU
            axis_features.append(torch.cat(feats, dim=0))

    # 3. STEP 2: Local Feature Fusion (LFF-AE)
    # Fusing features from each sensor unit independently [cite: 41, 168]
    ckpt_lff = f"{ckpt_dir}/lff_ae.pth"
    lff_input_all = torch.cat(axis_features, dim=1)
    model_lff = LFF_AE(input_channels=6*256).to(device) # C4 is fixed to 256 [cite: 353]

    if os.path.exists(ckpt_lff):
        print("Loading checkpoint for LFF-AE")
        model_lff.load_state_dict(torch.load(ckpt_lff, map_location=device))
    else:
        print("\n--- Training Local Feature Fusion (LFF-AE) ---")
        lff_input_train = lff_input_all[train_indices]
        lff_loader = DataLoader(TensorDataset(lff_input_train), batch_size=64, shuffle=True)
        model_lff = train_fusion_block(model_lff, lff_loader, device)
        torch.save(model_lff.state_dict(), ckpt_lff)
    
    model_lff.eval()
    with torch.no_grad():
        _, lff_feat_all = model_lff(lff_input_all.to(device))

    # 4. STEP 3: Global Feature Fusion (GFF-AE)
    # Creating a unique feature set from all sensor units [cite: 9, 196]
    ckpt_gff = f"{ckpt_dir}/gff_ae.pth"
    model_gff = GFF_AE(num_sensors=1).to(device) # num_sensors depends on your dataset

    if os.path.exists(ckpt_gff):
        print("Loading checkpoint for GFF-AE")
        model_gff.load_state_dict(torch.load(ckpt_gff, map_location=device))
    else:
        print("\n--- Training Global Feature Fusion (GFF-AE) ---")
        gff_input_train = lff_feat_all[train_indices]
        gff_loader = DataLoader(TensorDataset(gff_input_train), batch_size=64, shuffle=True)
        model_gff = train_fusion_block(model_gff, gff_loader, device)
        torch.save(model_gff.state_dict(), ckpt_gff)

    # 5. Final Feature Extraction
    print("\n--- Final features extraction ---")
    model_gff.eval()
    with torch.no_grad():
        _, final_features = model_gff(lff_feat_all.to(device))
    
    # Global Average Pooling to flatten spatial dimensions [cite: 37]
    final_features_flat = torch.mean(final_features, dim=2).cpu().numpy()
    df_results = pd.concat([metadata, pd.DataFrame(final_features_flat)], axis=1)
    df_results.to_csv("posturalInstability/data/extracted_huf_features.csv", index=False)
    print("Feature extraction completed and saved.")

if __name__ == "__main__":
    main()