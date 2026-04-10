import torch
import numpy as np
import pandas as pd
import os
import gc # Garbage collector
from torch.utils.data import DataLoader, TensorDataset
from model import DR_SAE, LFF_AE, GFF_AE
from train import train_stacked_dr_sae, train_fusion_block

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_dir = "posturalInstability/huf/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    metadata = pd.read_csv("posturalInstability/data/windowed_data/metadata.csv")
    subjects = metadata['subjectID'].unique()
    np.random.shuffle(subjects)
    
    train_subjects = subjects[:int(0.7 * len(subjects))]
    train_indices = metadata[metadata['subjectID'].isin(train_subjects)].index.tolist()
    
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # 1. STEP 1: DR-SAE Training & Feature Extraction to Disk
    for col in sensor_cols:
        ckpt_path = f"{ckpt_dir}/dr_sae_{col}.pth"
        feat_path = f"{ckpt_dir}/features_{col}.pt" # Save as PyTorch tensor file
        
        # Load raw data for this axis
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
            del train_data # Free RAM
        
        # Extract features and save to disk immediately
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
                del feats_list, all_axis_feats # Free RAM
        
        # CLEANUP before next axis
        del data, full_data, model_dr
        gc.collect()
        torch.cuda.empty_cache()

    # 2. STEP 2: Local Feature Fusion (LFF-AE)
    ckpt_lff = f"{ckpt_dir}/lff_ae.pth"
    
    # Load all axis features from disk only when needed
    print("\nLoading axis features for LFF-AE...")
    axis_features = [torch.load(f"{ckpt_dir}/features_{col}.pt") for col in sensor_cols]
    lff_input_all = torch.cat(axis_features, dim=1)
    del axis_features # Free individual tensors
    gc.collect()

    model_lff = LFF_AE(input_channels=6*256).to(device)

    if os.path.exists(ckpt_lff):
        print("Loading checkpoint for LFF-AE")
        model_lff.load_state_dict(torch.load(ckpt_lff, map_location=device))
    else:
        print("\n--- Training Local Feature Fusion (LFF-AE) ---")
        lff_input_train = lff_input_all[train_indices]
        lff_loader = DataLoader(TensorDataset(lff_input_train), batch_size=64, shuffle=True)
        model_lff = train_fusion_block(model_lff, lff_loader, device, block_name="LFF")
        torch.save(model_lff.state_dict(), ckpt_lff)
        del lff_input_train
    
    model_lff.eval()
    with torch.no_grad():
        # Process in batches to avoid GPU OOM
        lff_loader_full = DataLoader(TensorDataset(lff_input_all), batch_size=256, shuffle=False)
        lff_feats_list = []
        for b in lff_loader_full:
            _, lff_feat = model_lff(b[0].to(device))
            lff_feats_list.append(lff_feat.cpu())
        lff_feat_all = torch.cat(lff_feats_list, dim=0)
        del lff_feats_list, lff_input_all
        gc.collect()

    # 3. STEP 3: Global Feature Fusion (GFF-AE)
    ckpt_gff = f"{ckpt_dir}/gff_ae.pth"
    model_gff = GFF_AE(num_sensors=1).to(device)

    if os.path.exists(ckpt_gff):
        print("Loading checkpoint for GFF-AE")
        model_gff.load_state_dict(torch.load(ckpt_gff, map_location=device))
    else:
        print("\n--- Training Global Feature Fusion (GFF-AE) ---")
        gff_input_train = lff_feat_all[train_indices]
        gff_loader = DataLoader(TensorDataset(gff_input_train), batch_size=64, shuffle=True)
        model_gff = train_fusion_block(model_gff, gff_loader, device, block_name="GFF")
        torch.save(model_gff.state_dict(), ckpt_gff)

    # 4. Final Extraction
    print("\n--- Final features extraction ---")
    model_gff.eval()
    with torch.no_grad():
        gff_loader_full = DataLoader(TensorDataset(lff_feat_all), batch_size=256, shuffle=False)
        final_list = []
        for b in gff_loader_full:
            _, out = model_gff(b[0].to(device))
            final_list.append(out.cpu())
        final_features = torch.cat(final_list, dim=0)

    final_features_flat = torch.mean(final_features, dim=2).numpy()
    df_results = pd.concat([metadata, pd.DataFrame(final_features_flat)], axis=1)
    df_results.to_csv("posturalInstability/data/extracted_huf_features.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()