import re

import torch
import numpy as np
import pandas as pd
import os
import gc 
from torch.utils.data import DataLoader, Dataset
from model import DR_SAE, LFF_AE
from train import train_stacked_dr_sae, train_fusion_block


def preprocess_axis_windows(raw_data, metadata, train_indices, axis_name):
    """Apply stance detrending and train-fitted z-score normalization."""
    data = raw_data.astype(np.float32, copy=True)

    # Mean-center only in stance windows (task 0/1): acc_y/acc_z and all gyros.
    stance_mask = metadata["taskID"].isin([0, 1]).to_numpy()
    should_center = axis_name in {"acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
    if should_center:
        stance_idx = np.where(stance_mask)[0]
        if stance_idx.size > 0:
            data[stance_idx] = data[stance_idx] - data[stance_idx].mean(axis=1, keepdims=True)

    # Fit normalization on train only, then apply to all splits.
    train_values = data[train_indices].reshape(-1)
    mean_train = float(train_values.mean())
    std_train = float(train_values.std())
    if std_train < 1e-8:
        std_train = 1.0

    data = (data - mean_train) / std_train
    stats = {
        "mean": mean_train,
        "std": std_train,
        "centered_in_stance": bool(should_center),
    }
    return data, stats

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
    preprocessing_stats = {}

    # --- STEP 1: DR-SAE Training & Feature Extraction ---
    for col in sensor_cols:
        ckpt_path = f"{ckpt_dir}/dr_sae_{col}.pth"
        feat_path = f"{ckpt_dir}/features_{col}.npy" 
        
        if not os.path.exists(feat_path):
            # Load raw data only when needed
            raw_data = np.load(f"posturalInstability/data/windowed_data/axes/{col}.npy")
            raw_data, col_stats = preprocess_axis_windows(raw_data, metadata, train_indices, col)
            preprocessing_stats[col] = col_stats
            model_dr = DR_SAE().to(device)

            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint for DR-SAE: {col}")
                model_dr.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                print(f"\n--- Training DR-SAE: {col} ---")
                train_data = torch.from_numpy(raw_data[train_indices]).float()
                train_loader = DataLoader(
                    torch.utils.data.TensorDataset(train_data),
                    batch_size=16,
                    shuffle=True,
                    pin_memory=(device.type == "cuda"),
                )
                model_dr = train_stacked_dr_sae(
                    model_dr,
                    train_loader,
                    device,
                    min_epochs=5,
                    max_epochs=40,
                    target_loss=0.005,
                    layer_batch_size=16,
                )
                torch.save(model_dr.state_dict(), ckpt_path)
                del train_data

            print(f"Extracting features for {col} to disk...")
            model_dr.eval()
            raw_tensor = torch.from_numpy(raw_data).float()
            full_loader = DataLoader(
                torch.utils.data.TensorDataset(raw_tensor),
                batch_size=128,
                shuffle=False,
                pin_memory=(device.type == "cuda"),
            )

            # Stream features directly to disk to avoid RAM spikes from list + concatenate.
            feature_mmap = None
            write_pos = 0
            with torch.inference_mode():
                for b in full_loader:
                    x = b[0].to(device, non_blocking=(device.type == "cuda"))
                    _, f = model_dr(x)
                    f_np = f.detach().cpu().numpy()

                    if feature_mmap is None:
                        out_shape = (raw_data.shape[0],) + f_np.shape[1:]
                        feature_mmap = np.lib.format.open_memmap(
                            feat_path,
                            mode="w+",
                            dtype=f_np.dtype,
                            shape=out_shape,
                        )

                    batch_n = f_np.shape[0]
                    feature_mmap[write_pos:write_pos + batch_n] = f_np
                    write_pos += batch_n

                    del x, f, f_np

            # Ensure file metadata is fully written.
            if feature_mmap is not None:
                feature_mmap.flush()

            del raw_data, raw_tensor, model_dr, feature_mmap
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"Features for {col} already exist. Skipping.")

    if preprocessing_stats:
        stats_df = pd.DataFrame.from_dict(preprocessing_stats, orient="index")
        stats_path = os.path.join(ckpt_dir, "preprocessing_stats.csv")
        stats_df.to_csv(stats_path)
        print(f"Saved preprocessing stats to: {stats_path}")

    # --- STEP 2: Local Feature Fusion (LFF-AE) ---
    ckpt_lff = f"{ckpt_dir}/lff_ae.pth"
    train_dataset = MmapLFFDataset(feature_paths, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0 )

    

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
    extract_loader = DataLoader(full_lff_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    final_list = []
    with torch.no_grad():
        for b in extract_loader:
            _, lff_feat = model_lff(b.to(device))
            # Temporal Global Average Pooling as per standard paper practices
            final_list.append(torch.mean(lff_feat, dim=2).cpu())
    
    final_features_flat = torch.cat(final_list, dim=0).numpy()
    
    # 3. Save Results
    df_results = pd.concat([metadata, pd.DataFrame(final_features_flat)], axis=1)

    # merge with clinical data
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clinical_folder = os.path.join(root, 'data', 'cleaned_data')
    clinical_data = pd.DataFrame()
    for file in os.listdir(clinical_folder):
        if file.endswith('clinical.csv') or file.endswith('clinical_data.csv'):
            df = pd.read_csv(os.path.join(clinical_folder, file), dtype={'subjectID': str})
            df['dataset'] = re.sub(r'_clinical(?:_data)?\.csv$', '', file)
            clinical_data = pd.concat([clinical_data, df], ignore_index=True)

    df_results = df_results.merge(clinical_data, on=['subjectID', 'dataset'], how='left')
    df_results.to_csv("posturalInstability/data/features.csv", index=False)
    
    print(f"Extraction completed.")

if __name__ == "__main__":
    main()