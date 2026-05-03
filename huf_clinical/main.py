import os
import gc
import re

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from model import DR_SAE, LFF_AE
from train import train_stacked_dr_sae, train_fusion_block, fine_tune_lff_clinical


def preprocess_axis_windows(raw_data, metadata, train_indices, axis_name):
    """Apply stance detrending and train-fitted z-score normalization."""
    data = raw_data.astype(np.float32, copy=True)

    stance_mask = metadata["taskID"].isin([0, 1]).to_numpy()
    should_center = axis_name in {"acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
    if should_center:
        stance_idx = np.where(stance_mask)[0]
        if stance_idx.size > 0:
            data[stance_idx] = data[stance_idx] - data[stance_idx].mean(axis=1, keepdims=True)

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
        self.data_views = [np.load(p, mmap_mode="r") for p in self.feature_paths]

        if self.indices is not None:
            self.length = len(self.indices)
        else:
            self.length = self.data_views[0].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        feats = [torch.from_numpy(view[real_idx].copy()).float() for view in self.data_views]
        return torch.cat(feats, dim=0)


class MmapLFFClinicalDataset(MmapLFFDataset):
    def __init__(self, feature_paths, targets, indices=None):
        super().__init__(feature_paths, indices=indices)
        self.targets = np.asarray(targets, dtype=np.float32)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        x = super().__getitem__(idx)
        y = self.targets[real_idx]
        has_target = not np.isnan(y)
        y_tensor = torch.tensor(0.0 if np.isnan(y) else y, dtype=torch.float32)
        return x, y_tensor, torch.tensor(has_target, dtype=torch.bool)


def load_clinical_data(root):
    clinical_folder = os.path.join(root, "data", "cleaned_data")
    clinical_data = pd.DataFrame(columns=["subjectID", "dataset"])

    for file in os.listdir(clinical_folder):
        if file.endswith("clinical.csv") or file.endswith("clinical_data.csv"):
            df = pd.read_csv(os.path.join(clinical_folder, file), dtype={"subjectID": str}, low_memory=False)
            df["subjectID"] = df["subjectID"].astype(str)
            df["dataset"] = re.sub(r"_clinical(?:_data)?\.csv$", "", file)
            clinical_data = pd.concat([clinical_data, df], ignore_index=True)

    if clinical_data.empty:
        return pd.DataFrame(columns=["subjectID", "dataset", "postural_stability"])

    clinical_data = clinical_data.groupby(["subjectID", "dataset"], as_index=False).first()
    return clinical_data


def build_window_targets(metadata, clinical_data, target_col="postural_stability"):
    keys = metadata[["subjectID", "dataset"]].copy()
    keys["subjectID"] = keys["subjectID"].astype(str)
    keys["dataset"] = keys["dataset"].astype(str)

    if target_col not in clinical_data.columns:
        return np.full(len(metadata), np.nan, dtype=np.float32)

    target_df = clinical_data[["subjectID", "dataset", target_col]].copy()
    merged = keys.merge(target_df, on=["subjectID", "dataset"], how="left")
    return pd.to_numeric(merged[target_col], errors="coerce").to_numpy(dtype=np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data", "windowed_data")
    checkpoint_dir = os.path.join(data_dir, "checkpoints")
    feature_dir = os.path.join(data_dir, "features")
    extracted_dir = os.path.join(data_dir, "extracted_features")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(extracted_dir, exist_ok=True)

    # Load train/val/test data from huf_clinical split
    X_train = np.load(f"{data_dir}/train/windows.npy")
    y_train = np.load(f"{data_dir}/train/labels.npy")
    meta_train = pd.read_csv(f"{data_dir}/train/metadata.csv", low_memory=False)
    
    X_val = np.load(f"{data_dir}/val/windows.npy") if os.path.exists(f"{data_dir}/val/windows.npy") else np.array([])
    y_val = np.load(f"{data_dir}/val/labels.npy") if os.path.exists(f"{data_dir}/val/labels.npy") else np.array([])
    meta_val = pd.read_csv(f"{data_dir}/val/metadata.csv", low_memory=False) if os.path.exists(f"{data_dir}/val/metadata.csv") else pd.DataFrame()
    
    X_test = np.load(f"{data_dir}/test/windows.npy") if os.path.exists(f"{data_dir}/test/windows.npy") else np.array([])
    y_test = np.load(f"{data_dir}/test/labels.npy") if os.path.exists(f"{data_dir}/test/labels.npy") else np.array([])
    meta_test = pd.read_csv(f"{data_dir}/test/metadata.csv", low_memory=False) if os.path.exists(f"{data_dir}/test/metadata.csv") else pd.DataFrame()
    
    # Combine all for preprocessing statistics (train indices are 0:len(X_train))
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    metadata = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
    metadata["subjectID"] = metadata["subjectID"].astype(str)
    metadata["dataset"] = metadata["dataset"].astype(str)
    window_targets = np.concatenate([y_train, y_val, y_test], axis=0).astype(np.float32)
    
    # Train indices for normalization statistics
    train_indices = np.arange(len(X_train))

    clinical_data = load_clinical_data(root)

    sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    feature_paths = [os.path.join(feature_dir, f"features_{col}.npy") for col in sensor_cols]
    preprocessing_stats = {}

    # Step 1: DR-SAE training and feature extraction per axis
    for axis_idx, col in enumerate(sensor_cols):
        ckpt_path = os.path.join(checkpoint_dir, f"dr_sae_{col}.pth")
        feat_path = os.path.join(feature_dir, f"features_{col}.npy")

        if not os.path.exists(feat_path):
            # Extract axis data from X_all windows (windows shape: [n_windows, n_timepoints, 6])
            raw_data = X_all[:, :, axis_idx].copy().astype(np.float32)
            raw_data, col_stats = preprocess_axis_windows(raw_data, metadata, train_indices, col)
            preprocessing_stats[col] = col_stats

            model_dr = DR_SAE().to(device)
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint for DR-SAE: {col}")
                model_dr.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                print(f"\n--- Training DR-SAE: {col} ---")
                train_data = torch.from_numpy(raw_data[train_indices]).float().unsqueeze(1)  # [n_samples, 1, timesteps]
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
            raw_tensor = torch.from_numpy(raw_data).float().unsqueeze(1)  # [n_samples, 1, timesteps]
            full_loader = DataLoader(
                torch.utils.data.TensorDataset(raw_tensor),
                batch_size=128,
                shuffle=False,
                pin_memory=(device.type == "cuda"),
            )

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

            if feature_mmap is not None:
                feature_mmap.flush()

            del raw_data, raw_tensor, model_dr, feature_mmap
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"Features for {col} already exist. Skipping.")

    if preprocessing_stats:
        stats_df = pd.DataFrame.from_dict(preprocessing_stats, orient="index")
        stats_path = os.path.join(extracted_dir, "preprocessing_stats.csv")
        stats_df.to_csv(stats_path)
        print(f"Saved preprocessing stats to: {stats_path}")

    # Step 2: LFF training
    ckpt_lff = os.path.join(checkpoint_dir, "lff_ae.pth")
    ckpt_lff_clinical = os.path.join(checkpoint_dir, "lff_ae_clinical.pth")
    train_dataset = MmapLFFDataset(feature_paths, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    model_lff = LFF_AE(input_channels=6 * 256).to(device)

    if os.path.exists(ckpt_lff):
        print("Loading checkpoint for LFF-AE")
        model_lff.load_state_dict(torch.load(ckpt_lff, map_location=device))
    else:
        print("\n--- Training Local Feature Fusion (LFF-AE) ---")
        model_lff = train_fusion_block(model_lff, train_loader, device, block_name="LFF")
        torch.save(model_lff.state_dict(), ckpt_lff)

    # Step 2B: Clinical-aware fine-tuning
    if os.path.exists(ckpt_lff_clinical):
        print("Loading checkpoint for Clinical-aware LFF")
        model_lff.load_state_dict(torch.load(ckpt_lff_clinical, map_location=device))
    else:
        train_clinical_dataset = MmapLFFClinicalDataset(feature_paths, targets=window_targets, indices=train_indices)
        train_clinical_loader = DataLoader(train_clinical_dataset, batch_size=16, shuffle=True, num_workers=0)
        model_lff = fine_tune_lff_clinical(
            model_lff,
            train_clinical_loader,
            device,
            alpha=0.2,
            lr=5e-4,
            max_epochs=20,
            min_epochs=8,
            patience=6,
        )
        torch.save(model_lff.state_dict(), ckpt_lff_clinical)

    # Final feature extraction
    print("\n--- Final features extraction from clinical-aware LFF-AE ---")
    model_lff.eval()

    full_lff_dataset = MmapLFFDataset(feature_paths)
    extract_loader = DataLoader(full_lff_dataset, batch_size=128, shuffle=False, num_workers=0)

    final_list = []
    with torch.no_grad():
        for b in extract_loader:
            _, lff_feat = model_lff(b.to(device))
            final_list.append(torch.mean(lff_feat, dim=2).cpu())

    final_features_flat = torch.cat(final_list, dim=0).numpy()

    df_results = pd.concat([metadata, pd.DataFrame(final_features_flat)], axis=1)
    df_results = df_results.merge(clinical_data, on=["subjectID", "dataset"], how="left")

    if "postural_stability_y" in df_results.columns:
        df_results["postural_stability"] = df_results["postural_stability_y"]
    elif "postural_stability_x" in df_results.columns:
        df_results["postural_stability"] = df_results["postural_stability_x"]
    elif "postural_stability" not in df_results.columns and "label" in df_results.columns:
        df_results["postural_stability"] = df_results["label"]

    combined_output_path = os.path.join(extracted_dir, "features_clinical_aware.csv")
    df_results.to_csv(combined_output_path, index=False)

    split_map = {
        "train": np.arange(0, len(X_train)),
        "val": np.arange(len(X_train), len(X_train) + len(X_val)),
        "test": np.arange(len(X_train) + len(X_val), len(X_train) + len(X_val) + len(X_test)),
    }
    for split_name, split_indices in split_map.items():
        split_dir = os.path.join(extracted_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        split_df = df_results.iloc[split_indices].reset_index(drop=True)
        split_csv_path = os.path.join(split_dir, "features_clinical_aware.csv")
        split_df.to_csv(split_csv_path, index=False)

    print(f"Clinical-aware extraction completed: {combined_output_path}")


if __name__ == "__main__":
    main()
