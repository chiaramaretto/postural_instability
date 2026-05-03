import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model_att import SubjectTaskAttentionClassifier
from train_att import evaluate_classifier, fit_classifier


@dataclass
class SplitPaths:
    train: str
    val: str
    test: str


class SubjectBagDataset(Dataset):
    def __init__(self, bags):
        self.bags = bags

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx]


def collate_subject_bags(batch):
    max_len = max(item["x"].shape[0] for item in batch)
    feat_dim = batch[0]["x"].shape[1]

    x = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    task_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    window_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    y = torch.zeros((len(batch),), dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        x[i, :n] = torch.from_numpy(np.asarray(item["x"], dtype=np.float32))
        task_ids[i, :n] = torch.from_numpy(np.asarray(item["task_ids"], dtype=np.int64))
        window_mask[i, :n] = True
        y[i] = int(item["y"])

    return {"x": x, "task_ids": task_ids, "window_mask": window_mask, "y": y}


def resolve_split_paths(root):
    base_dir = os.path.join(root, "posturalInstability", "huf_clinical", "data", "windowed_data", "extracted_features")
    return SplitPaths(
        train=os.path.join(base_dir, "train", "features_clinical_aware.csv"),
        val=os.path.join(base_dir, "val", "features_clinical_aware.csv"),
        test=os.path.join(base_dir, "test", "features_clinical_aware.csv"),
    )


def resolve_label_column(df):
    for column in ["postural_stability", "postural_stability_y", "postural_stability_x", "label"]:
        if column in df.columns:
            return column
    raise ValueError("No label column found. Expected postural_stability or label.")


def build_subject_bags(features_df):
    features_df = features_df.copy()
    features_df["subjectID"] = features_df["subjectID"].astype(str)
    features_df["dataset"] = features_df["dataset"].astype(str)

    feature_cols = [column for column in features_df.columns if str(column).isdigit()]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in the HUF output CSV.")
    if "taskID" not in features_df.columns:
        raise ValueError("Column taskID not found in the HUF output CSV.")

    label_col = resolve_label_column(features_df)
    bags = []
    for (subject_id, dataset), group in features_df.groupby(["subjectID", "dataset"]):
        y_value = pd.to_numeric(group[label_col], errors="coerce").iloc[0]
        if pd.isna(y_value):
            continue

        y_int = int(np.clip(np.round(float(y_value)), 0, 4))
        bags.append(
            {
                "subjectID": str(subject_id),
                "dataset": str(dataset),
                "x": group[feature_cols].to_numpy(dtype=np.float32),
                "task_ids": pd.to_numeric(group["taskID"], errors="coerce").fillna(0).astype(int).to_numpy(),
                "y": y_int,
            }
        )

    if not bags:
        raise ValueError("No subject-level bags could be built from the provided features.")

    return bags, feature_cols


def compute_class_weights(bags, num_classes):
    labels = np.array([bag["y"] for bag in bags], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()
    return weights


def make_loader(bags, batch_size, shuffle):
    dataset = SubjectBagDataset(bags)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_subject_bags)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_paths = resolve_split_paths(repo_root)

    for split_name, path in [("train", split_paths.train), ("val", split_paths.val), ("test", split_paths.test)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {split_name} features file: {path}")

    train_df = pd.read_csv(split_paths.train, low_memory=False)
    val_df = pd.read_csv(split_paths.val, low_memory=False)
    test_df = pd.read_csv(split_paths.test, low_memory=False)

    train_bags, feature_cols = build_subject_bags(train_df)
    val_bags, _ = build_subject_bags(val_df)
    test_bags, _ = build_subject_bags(test_df)

    all_labels = np.array([bag["y"] for bag in train_bags + val_bags + test_bags], dtype=np.int64)
    num_classes = int(all_labels.max()) + 1
    num_tasks = int(max(
        train_df["taskID"].max(),
        val_df["taskID"].max(),
        test_df["taskID"].max(),
    )) + 1

    train_loader = make_loader(train_bags, batch_size=8, shuffle=True)
    val_loader = make_loader(val_bags, batch_size=8, shuffle=False)
    test_loader = make_loader(test_bags, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train subjects: {len(train_bags)} | Val subjects: {len(val_bags)} | Test subjects: {len(test_bags)}")
    print(f"Feature dim: {len(feature_cols)} | Tasks: {num_tasks} | Classes: {num_classes}")

    model = SubjectTaskAttentionClassifier(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_tasks=num_tasks,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    class_weights = compute_class_weights(train_bags, num_classes)
    checkpoint_dir = os.path.join(repo_root, "posturalInstability", "attention", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "patient_level_attention_classifier.pth")

    model = fit_classifier(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=80,
        patience=12,
        min_delta=1e-3,
        scheduler_patience=5,
        scheduler_factor=0.5,
        min_lr=1e-5,
        class_weights=class_weights,
        best_path=best_path,
    )

    test_metrics = evaluate_classifier(model, test_loader, device)
    print(
        "Test metrics | "
        f"Accuracy: {test_metrics['accuracy']:.4f} | "
        f"F1 macro: {test_metrics['f1_macro']:.4f} | "
        f"AUC macro: {test_metrics['auc_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
