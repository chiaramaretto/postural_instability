from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from model import GRUCNNEnsembleClassifier
from train import fit_model


def find_repo_root():
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Unable to locate repository root")


def load_window_artifacts(root: Path):
    path = root / "cnn_gru" / "data" / "windowed"
    windows_path = path / "windows.npy"
    meta_path = path / "metadata.csv"
    if not windows_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Run cnn_gru/preprocessing.ipynb to create windowed artifacts")
    windows = np.load(windows_path)
    meta = pd.read_csv(meta_path, dtype={"subjectID": str, "dataset": str})
    return windows.astype(np.float32), meta


def load_clinical(root: Path):
    folder = root / "data" / "cleaned_data"
    frames = []
    if not folder.exists():
        return pd.DataFrame(columns=["subjectID", "dataset", "postural_stability"])
    for p in folder.iterdir():
        if p.name.endswith("clinical.csv") or p.name.endswith("clinical_data.csv"):
            df = pd.read_csv(p, dtype={"subjectID": str})
            df["dataset"] = p.name.replace("_clinical.csv", "").replace("_clinical_data.csv", "")
            frames.append(df[["subjectID", "dataset", "postural_stability"]])
    if not frames:
        return pd.DataFrame(columns=["subjectID", "dataset", "postural_stability"])
    allf = pd.concat(frames, ignore_index=True)
    return allf.groupby(["subjectID", "dataset"], as_index=False).first()


def build_labels(metadata: pd.DataFrame, clinical: pd.DataFrame):
    merged = metadata.merge(clinical, on=["subjectID", "dataset"], how="left")
    targets = pd.to_numeric(merged["postural_stability"], errors="coerce")
    # map to integer classes 0-4
    classes = np.clip(np.rint(targets).fillna(-1).astype(int), 0, 4)
    valid_mask = targets.notna()
    return classes.to_numpy(), valid_mask.to_numpy()


def minmax_scale(train_windows: np.ndarray, others: np.ndarray):
    # train_windows: (n_train, seq_len, channels)
    mins = train_windows.min(axis=(0, 1), keepdims=True)
    maxs = train_windows.max(axis=(0, 1), keepdims=True)
    denom = (maxs - mins).clip(min=1e-6)
    train_scaled = (train_windows - mins) / denom
    others_scaled = (others - mins) / denom
    return train_scaled.astype(np.float32), others_scaled.astype(np.float32), {"min": mins, "max": maxs}


def oversample_to_N(windows, classes, N=60, k=2, random_state=42):
    rng = np.random.default_rng(random_state)
    unique, counts = np.unique(classes, return_counts=True)
    all_windows = [windows]
    all_classes = [classes]

    for c in range(5):
        mask = classes == c
        cls_windows = windows[mask]
        cls_classes = classes[mask]
        if len(cls_windows) == 0:
            continue
        if len(cls_windows) >= N:
            continue
        # flatten time+channels for distance
        flat = cls_windows.reshape(len(cls_windows), -1)
        n_neighbors = min(k + 1, len(cls_windows))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(flat)
        neigh = nn.kneighbors(flat, return_distance=False)
        synth = []
        synth_c = []
        while len(synth) < (N - len(cls_windows)):
            i = int(rng.integers(0, len(cls_windows)))
            candidates = neigh[i]
            candidates = candidates[candidates != i]
            if len(candidates) == 0:
                j = i
            else:
                j = int(rng.choice(candidates[:k]))
            alpha = float(rng.random())
            s = cls_windows[i] + alpha * (cls_windows[j] - cls_windows[i])
            synth.append(s)
            synth_c.append(c)
        if synth:
            all_windows.append(np.stack(synth, axis=0).astype(np.float32))
            all_classes.append(np.array(synth_c, dtype=int))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_classes, axis=0)
    perm = np.random.default_rng(42).permutation(len(y))
    return X[perm], y[perm]


def make_loader(windows: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X = torch.from_numpy(windows).float()
    y = torch.from_numpy(targets).long()
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def main():
    root = find_repo_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    windows, metadata = load_window_artifacts(root)
    clinical = load_clinical(root)

    classes, valid_mask = build_labels(metadata, clinical)
    if not valid_mask.any():
        raise RuntimeError("No labeled windows found.")

    windows = windows[valid_mask]
    classes = classes[valid_mask]
    metadata = metadata.loc[valid_mask].reset_index(drop=True)

    # subject-wise split
    groups = metadata["subjectID"].astype(str) + "__" + metadata["dataset"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))

    train_windows = windows[train_idx]
    train_classes = classes[train_idx]
    test_windows = windows[test_idx]
    test_classes = classes[test_idx]

    # normalization: min-max on training set
    train_windows, test_windows, stats = minmax_scale(train_windows, test_windows)

    # oversample to N=60 per class using k=2
    train_windows, train_classes = oversample_to_N(train_windows, train_classes, N=60, k=2, random_state=42)

    train_loader = make_loader(train_windows, train_classes, batch_size=64, shuffle=True)
    test_loader = make_loader(test_windows, test_classes, batch_size=128, shuffle=False)

    n_channels = train_windows.shape[-1]
    model = GRUCNNEnsembleClassifier(input_channels=n_channels)
    model.to(device)

    ckpt_dir = root / "cnn_gru" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "grucnn_classifier.pt"

    model = fit_model(model, train_loader, test_loader, device, max_epochs=200, patience=20, best_path=str(best_path))

    # final evaluation
    from train import evaluate_model
    metrics = evaluate_model(model, test_loader, device, aggregate_by_group=False)
    print(f"Final validation accuracy: {metrics['acc']:.4f} on {metrics['n']} samples")


if __name__ == '__main__':
    main()
