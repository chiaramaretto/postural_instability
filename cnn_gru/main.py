import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from model import CnnGru
from train import fit_model, predict
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Paper configuration
TASK_CONFIG = {
    "static": {"batch_size": 32, "lr": 1e-4}, # Task 0 + 1
    2: {"batch_size": 64, "lr": 1e-4},        # Task 2
}

def SMOTE_augmentation(X, y, target_count=200, k_neighbors=2):
    unique_labels = np.unique(y)
    aug_X, aug_y = [], []

    for label in unique_labels:
        idx = np.where(y == label)[0]
        cls_X = X[idx]
        
        if len(cls_X) >= target_count:
            chosen = np.random.choice(len(cls_X), target_count, replace=False)
            aug_X.append(cls_X[chosen])
            aug_y.append(np.full(target_count, label))
            continue

        aug_X.append(cls_X)
        aug_y.append(y[idx])
        
        num_to_add = target_count - len(cls_X)
        flat_X = cls_X.reshape(len(cls_X), -1)
        nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(cls_X)))
        nn.fit(flat_X)
        knns = nn.kneighbors(flat_X, return_distance=False)

        synth_samples = []
        for _ in range(num_to_add):
            i = np.random.randint(0, len(cls_X))
            neighbor_idx = np.random.choice(knns[i][1:]) if len(knns[i]) > 1 else i
            
            alpha = np.random.random()
            diff = cls_X[neighbor_idx] - cls_X[i]
            synth_samples.append(cls_X[i] + alpha * diff)
            
        aug_X.append(np.stack(synth_samples))
        aug_y.append(np.full(num_to_add, label))

    return np.concatenate(aug_X), np.concatenate(aug_y)

def jittering_augmentation(X, y, target_count=200, sigma=0.05):
    unique_labels = np.unique(y)
    aug_X, aug_y = [], []

    for label in unique_labels:
        idx = np.where(y == label)[0]
        cls_X = X[idx]
        
        if len(cls_X) >= target_count:
            chosen = np.random.choice(len(cls_X), target_count, replace=False)
            aug_X.append(cls_X[chosen])
            aug_y.append(np.full(target_count, label))
            continue

        aug_X.append(cls_X)
        aug_y.append(y[idx])
        
        num_to_add = target_count - len(cls_X)
        noise = np.random.normal(0, sigma, size=(num_to_add, *cls_X.shape[1:]))
        synth_samples = cls_X[np.random.choice(len(cls_X), num_to_add)] + noise
        aug_X.append(synth_samples)
        aug_y.append(np.full(num_to_add, label))

    return np.concatenate(aug_X), np.concatenate(aug_y)

def flipping_augmentation(X, y, target_count=200):
    unique_labels = np.unique(y)
    aug_X, aug_y = [], []

    for label in unique_labels:
        idx = np.where(y == label)[0]
        cls_X = X[idx]
        
        if len(cls_X) >= target_count:
            chosen = np.random.choice(len(cls_X), target_count, replace=False)
            aug_X.append(cls_X[chosen])
            aug_y.append(np.full(target_count, label))
            continue

        aug_X.append(cls_X)
        aug_y.append(y[idx])
        
        num_to_add = target_count - len(cls_X)
        flipped_samples = cls_X[np.random.choice(len(cls_X), num_to_add)] * np.array([[-1, -1, -1, 1, 1, 1]])
        aug_X.append(flipped_samples)
        aug_y.append(np.full(num_to_add, label))

    return np.concatenate(aug_X), np.concatenate(aug_y)

def stretching_augmentation(X, y, target_count=200, stretch_factor_range=(0.8, 1.2)):
    unique_labels = np.unique(y)
    aug_X, aug_y = [], []

    for label in unique_labels:
        idx = np.where(y == label)[0]
        cls_X = X[idx]

        if len(cls_X) >= target_count:
            chosen = np.random.choice(len(cls_X), target_count, replace=False)
            aug_X.append(cls_X[chosen])
            aug_y.append(np.full(target_count, label))
            continue

        aug_X.append(cls_X)
        aug_y.append(y[idx])

        num_to_add = target_count - len(cls_X)
        base_time = np.arange(cls_X.shape[1], dtype=np.float32)
        synth_samples = []

        for _ in range(num_to_add):
            sample_idx = np.random.randint(len(cls_X))
            sample = cls_X[sample_idx]
            stretch_factor = np.random.uniform(*stretch_factor_range)

            stretched_len = max(2, int(round(sample.shape[0] * stretch_factor)))
            stretched_time = np.linspace(0, sample.shape[0] - 1, stretched_len, dtype=np.float32)
            stretched = np.stack(
                [np.interp(stretched_time, base_time, sample[:, channel]) for channel in range(sample.shape[1])],
                axis=-1,
            )

            restore_time = np.linspace(0, stretched_len - 1, sample.shape[0], dtype=np.float32)
            restored = np.stack(
                [np.interp(restore_time, np.arange(stretched_len, dtype=np.float32), stretched[:, channel]) for channel in range(sample.shape[1])],
                axis=-1,
            )
            synth_samples.append(restored)

        aug_X.append(np.stack(synth_samples))
        aug_y.append(np.full(num_to_add, label))

    return np.concatenate(aug_X), np.concatenate(aug_y)

def augmentation(X, y, target_count=200):
    # randomly pick one or more augmentation techniques to apply
    aug_functions = [SMOTE_augmentation, jittering_augmentation, flipping_augmentation, stretching_augmentation]
    X_aug, y_aug = X.copy(), y.copy()
    for aug_func in np.random.choice(aug_functions, size=np.random.randint(1, len(aug_functions) + 1), replace=False):
        X_aug, y_aug = aug_func(X_aug, y_aug, target_count=target_count)
    return X_aug, y_aug

def get_subject_data(windows, labels, metadata):
    subject_keys = metadata.apply(lambda r: (r['dataset'], str(r['subjectID']).strip()), axis=1)
    subj_to_idx = {}
    for idx, key in enumerate(subject_keys):
        subj_to_idx.setdefault(tuple(key), []).append(idx)
    
    unique_subjects = list(subj_to_idx.keys())
    subj_labels = [int(pd.Series(labels[subj_to_idx[k]]).mode().iloc[0]) for k in unique_subjects]
    return np.array(unique_subjects), np.array(subj_labels), subj_to_idx

def run_experiment(name, X_task, y_task, meta_task):
    print(f"\n" + "="*40 + f"\nTASK: {name}\n" + "="*40)
    subjects, subj_labels, subj_to_idx = get_subject_data(X_task, y_task, meta_task)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    config = TASK_CONFIG.get(name if name == "static" else 2, {"batch_size": 32, "lr": 5e-4})

    # TensorFlow device selection
    device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

    for fold, (train_subj_idx, test_subj_idx) in enumerate(skf.split(subjects, subj_labels)):
        # 1. Subject Split
        train_indices = [idx for s_i in train_subj_idx for idx in subj_to_idx[tuple(subjects[s_i])]]
        test_indices = [idx for s_i in test_subj_idx for idx in subj_to_idx[tuple(subjects[s_i])]]
        
        X_train, y_train = X_task[train_indices], y_task[train_indices]
        X_test, y_test = X_task[test_indices], y_task[test_indices]

        # 2. Augmentation (Training set only)
        print("class distribution before augmentation:", np.bincount(y_train))
        target = np.max(np.bincount(y_train))
        #mix of augmentations to reach target count per class
        X_train, y_train = augmentation(X_train, y_train, target_count=target)

        print("class distribution after augmentation:", np.bincount(y_train))

        perm = np.random.permutation(len(y_train))
        X_train, y_train = X_train[perm], y_train[perm]

        # Normalize each window individually to remove per-window offsets and scale
        train_mean = X_train.mean(axis=1, keepdims=True)
        train_std = X_train.std(axis=1, keepdims=True)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        X_train = (X_train - train_mean) / train_std

        # Normalize test windows independently (per-window)
        test_mean = X_test.mean(axis=1, keepdims=True)
        test_std = X_test.std(axis=1, keepdims=True)
        test_std = np.where(test_std < 1e-6, 1.0, test_std)
        X_test = (X_test - test_mean) / test_std

        # 4. Training with TensorFlow
        # Initialize Keras model (ensure CnnGru returns a tf.keras.Model)
        model = CnnGru(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=4)

        model, _, _ = fit_model(
            model, X_train, y_train, device_name,
            batch_size=config["batch_size"], max_epochs=500, patience=20,
            X_val=X_test, y_val=y_test, lr=config["lr"]
        )

        y_pred = predict(model, X_test, device_name, batch_size=config["batch_size"])
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    print(f"\n{name} Average Accuracy: {np.mean(fold_accuracies):.4f}")
    return np.mean(fold_accuracies)

def main():
    # Load paths
    data_path = "posturalInstability/cnn_gru/data/windowed_data/"
    windows = np.load(os.path.join(data_path, "windows.npy"))
    labels = np.load(os.path.join(data_path, "labels.npy"))
    metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))

    # Label merging (Class 4 into 3)
    labels = np.where(labels == 4, 3, labels)
    
    results = {}
    static_mask = metadata['taskID'].isin([0, 1])
    if static_mask.any():
        results["Static (0+1)"] = run_experiment("static", windows[static_mask], labels[static_mask], metadata[static_mask].reset_index(drop=True))

    dynamic_mask = metadata['taskID'] == 2
    if dynamic_mask.any():
        results["Dynamic (2)"] = run_experiment("dynamic", windows[dynamic_mask], labels[dynamic_mask], metadata[dynamic_mask].reset_index(drop=True))

    print("\nSUMMARY:\n" + "\n".join([f"{k}: {v:.4f}" for k, v in results.items()]))
    if results: 
        print(f"Overall Accuracy: {np.mean(list(results.values())):.4f}")

if __name__ == "__main__":
    main()