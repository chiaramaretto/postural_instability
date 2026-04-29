import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from model import CnnGru
from train import fit_model, predict, print_confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


def print_confusion_matrix_by_dataset(y_true, y_pred, datasets, class_labels):
    datasets = np.asarray(datasets)
    unique_datasets = pd.Series(datasets).dropna().unique()

    print("\nConfusion matrix per dataset:")
    for dataset in unique_datasets:
        mask = datasets == dataset
        if not np.any(mask):
            continue

        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=class_labels)
        print(f"\nDataset: {dataset} | n={int(mask.sum())}")
        print(cm)


def main():
    windows = np.load("posturalInstability/cnn_gru/data/windowed_data/windows.npy")
    labels = np.load("posturalInstability/cnn_gru/data/windowed_data/labels.npy")
    metadata = pd.read_csv("posturalInstability/cnn_gru/data/windowed_data/metadata.csv")

    # print class distribution before oversampling
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution before oversampling:")
    for u, c in zip(unique, counts):    
        print(f"Class {u}: {c} samples")

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        windows,
        labels,
        metadata,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )


    # print class distribution after oversampling
    unique, counts = np.unique(y_train, return_counts=True)
    print("Class distribution after oversampling:")
    for u, c in zip(unique, counts):    
        print(f"Class {u}: {c} samples")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnGru(input_channels=6).to(device)
    
    # 4. Fit
    m, acc = fit_model(model, X_train, y_train, device, batch_size=32, max_epochs=500, patience=20)
    print(f"Best validation accuracy: {acc:.4f}")

    # Final test evaluation
    y_pred = predict(m, X_test, device, batch_size=32)
    print_confusion_matrix(y_test, y_pred, labels=np.unique(labels))
    print_confusion_matrix_by_dataset(y_test, y_pred, meta_test["dataset"].to_numpy(), np.unique(labels))

if __name__ == "__main__":
    main()