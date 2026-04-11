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

    metadata = pd.read_csv("posturalInstability/data/windowed_data/metadata.csv")
    subjects = metadata['subjectID'].unique()
    np.random.shuffle(subjects)
    
    train_subjects = subjects[:int(0.7 * len(subjects))]
    train_indices = metadata[metadata['subjectID'].isin(train_subjects)].index.tolist()
    
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # --- STEP 1: DR-SAE Training & Feature Extraction con Pooling ---
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
                    # RIDUZIONE TEMPORALE: Passiamo da [Batch, 256, 256] a [Batch, 256, 1]
                    # Questo riduce il peso del file di 256 volte!
                    f_pooled = torch.mean(f, dim=2, keepdim=True) 
                    feats_list.append(f_pooled.cpu())
                all_axis_feats = torch.cat(feats_list, dim=0)
                torch.save(all_axis_feats, feat_path)
                del feats_list, all_axis_feats 
        
        del data, full_data, model_dr
        gc.collect()
        torch.cuda.empty_cache()

    # --- STEP 2: Local Feature Fusion (LFF-AE) ---
    ckpt_lff = f"{ckpt_dir}/lff_ae.pth"
    print("\nLoading axis features for LFF-AE (Memory Efficient)...")
    
    num_samples = len(metadata)
    # Ora carichiamo feature compatte: [Campioni, 1536, 1]
    lff_input_all = torch.empty((num_samples, 6 * 256, 1), dtype=torch.float32)

    for i, col in enumerate(sensor_cols):
        feat_path = f"{ckpt_dir}/features_{col}.pt"
        axis_feat = torch.load(feat_path, map_location='cpu')
        lff_input_all[:, i*256 : (i+1)*256, :] = axis_feat
        del axis_feat
        gc.collect()

    model_lff = LFF_AE(input_channels=6*256).to(device)

    if os.path.exists(ckpt_lff):
        print("Loading checkpoint for LFF-AE")
        model_lff.load_state_dict(torch.load(ckpt_lff, map_location=device))
    else:
        print("\n--- Training LFF-AE ---")
        from torch.utils.data import Subset
        train_subset = Subset(TensorDataset(lff_input_all), train_indices)
        lff_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        model_lff = train_fusion_block(model_lff, lff_loader, device, block_name="LFF")
        torch.save(model_lff.state_dict(), ckpt_lff)
        del train_subset, lff_loader

    # --- FINAL EXTRACTION ---
    print("\n--- Final features extraction ---")
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

    # GAP finale per ottenere il vettore a 256 dimensioni
    final_features_flat = torch.mean(final_features, dim=2).squeeze().numpy()
    
    df_results = pd.concat([metadata, pd.DataFrame(final_features_flat)], axis=1)
    df_results.to_csv("posturalInstability/data/extracted_huf_features.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()