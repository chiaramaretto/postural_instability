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


def oversample(windows, labels, target_count=800, k_neighbors=3):

    unique_labels = np.unique(labels)
    new_windows = []
    new_labels = []
    
    rng = np.random.default_rng(42)

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        cls_windows = windows[idx]
        
        # Se la classe ha già abbastanza campioni, la usiamo così com'è 
        # o facciamo un leggero downsampling se vogliamo pareggiare a target_count
        if len(cls_windows) >= target_count:
            chosen_idx = rng.choice(idx, size=target_count, replace=False)
            new_windows.append(windows[chosen_idx])
            new_labels.append(labels[chosen_idx])
            continue

        new_windows.append(cls_windows) # Teniamo gli originali
        new_labels.append(labels[idx])
        
        flat_windows = cls_windows.reshape(len(cls_windows), -1)
        nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(cls_windows)), metric="euclidean")
        nn.fit(flat_windows)
        knns = nn.kneighbors(flat_windows, return_distance=False)

        synth_windows = []
        num_to_add = target_count - len(cls_windows)
        
        for _ in range(num_to_add):

            i = rng.integers(0, len(cls_windows))
            neighbor_idx = rng.choice(knns[i][1:]) # Escludiamo se stesso
            
            # Interpolazione lineare (SMOTE): crea una finestra "in mezzo" alle due
            alpha = rng.random()
            synthetic_sample = cls_windows[i] + alpha * (cls_windows[neighbor_idx] - cls_windows[i])
            
            # Aggiungiamo un leggero Jittering (rumore) come suggerito per la robustezza
            noise = rng.normal(0, 0.001, synthetic_sample.shape)
            synth_windows.append(synthetic_sample + noise)
            
        new_windows.append(np.stack(synth_windows))
        new_labels.append(np.full(num_to_add, label))

    return np.concatenate(new_windows), np.concatenate(new_labels)

def main():
    windows = np.load("posturalInstability/cnn_gru/data/windowed_data/windows.npy")
    labels = np.load("posturalInstability/cnn_gru/data/windowed_data/labels.npy")
    metadata = pd.read_csv("posturalInstability/cnn_gru/data/windowed_data/metadata.csv")

    # print class distribution before oversampling
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution before oversampling:")
    for u, c in zip(unique, counts):    
        print(f"Class {u}: {c} samples")

    X_train_raw, X_test, y_train_raw, y_test, meta_train, meta_test = train_test_split(
        windows,
        labels,
        metadata,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    X_train, y_train = oversample(X_train_raw, y_train_raw, target_count=800)

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