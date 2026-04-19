import torch
import numpy as np
import pandas as pd
import os
import gc 
from torch.utils.data import DataLoader, Dataset, TensorDataset
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

class MmapLFFDataset(Dataset):
    def __init__(self, feature_paths, indices=None):
        self.feature_paths = [p.replace(".pt", ".npy") for p in feature_paths]
        self.indices = indices
        self.data_views = [np.load(p, mmap_mode='r') for p in self.feature_paths]
        self.length = len(self.indices) if self.indices is not None else self.data_views[0].shape[0]

    def __len__(self): return self.length

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        feats = [torch.from_numpy(view[real_idx].copy()).float() for view in self.data_views]
        return torch.cat(feats, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = "posturalInstability/huf/checkpoints_clinical" # Nuova cartella per non sovrascrivere
    os.makedirs(ckpt_dir, exist_ok=True)

    metadata = pd.read_csv("posturalInstability/data/windowed_data/metadata.csv")
    train_indices = metadata[metadata['subjectID'].isin(metadata['subjectID'].unique()[:int(0.7*len(metadata['subjectID'].unique()))])].index.tolist()
    
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    feature_paths = [f"{ckpt_dir}/clinical_feat_{col}.npy" for col in sensor_cols]
    preprocessing_stats = {}

    for col in sensor_cols:
        feat_path = f"{ckpt_dir}/clinical_feat_{col}.npy" 
        if not os.path.exists(feat_path):
            raw_data = np.load(f"posturalInstability/data/windowed_data/axes/{col}.npy")
            raw_data, col_stats = preprocess_axis_windows(raw_data, metadata, train_indices, col)
            preprocessing_stats[col] = col_stats
            model_dr = DR_SAE().to(device)
            model_dr = train_stacked_dr_sae(model_dr, DataLoader(TensorDataset(torch.from_numpy(raw_data[train_indices]).float()), batch_size=16, shuffle=True), device)
            
            # Extraction
            model_dr.eval()
            feature_mmap = None
            with torch.no_grad():
                for i in range(0, len(raw_data), 128):
                    batch = torch.from_numpy(raw_data[i:i+128]).float().to(device)
                    _, f = model_dr(batch)
                    f_np = f.cpu().numpy()
                    if feature_mmap is None:
                        feature_mmap = np.lib.format.open_memmap(feat_path, mode="w+", dtype=f_np.dtype, shape=(len(raw_data), 32, f_np.shape[2]))
                    feature_mmap[i:i+len(f_np)] = f_np
            feature_mmap.flush()
            del raw_data, model_dr
            gc.collect()

    if preprocessing_stats:
        stats_df = pd.DataFrame.from_dict(preprocessing_stats, orient="index")
        stats_path = os.path.join(ckpt_dir, "preprocessing_stats.csv")
        stats_df.to_csv(stats_path)
        print(f"Saved preprocessing stats to: {stats_path}")

    # Fusion training (Input channels: 6 sensors * 32 clinical features)
    model_lff = LFF_AE(input_channels=6*32, c4_dim=32).to(device)
    train_loader = DataLoader(MmapLFFDataset(feature_paths, train_indices), batch_size=16, shuffle=True)
    model_lff = train_fusion_block(model_lff, train_loader, device, block_name="ClinicalLFF")

    # Final Average Pooled Features for Classifier
    model_lff.eval()
    final_list = []
    extract_loader = DataLoader(MmapLFFDataset(feature_paths), batch_size=128, shuffle=False)
    with torch.no_grad():
        for b in extract_loader:
            _, lff_feat = model_lff(b.to(device))
            final_list.append(torch.mean(lff_feat, dim=2).cpu()) # GAP per stabilità temporale
    
    final_features_flat = torch.cat(final_list, dim=0).numpy()
    pd.concat([metadata, pd.DataFrame(final_features_flat)], axis=1).to_csv("posturalInstability/data/clinical_huf_features.csv", index=False)
    print("✅ Clinical Features saved to: posturalInstability/data/clinical_huf_features.csv")

if __name__ == "__main__": main()