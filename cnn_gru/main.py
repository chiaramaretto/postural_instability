import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
from model import BagCnnGru
from train import fit_model, predict
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Paper configuration
TASK_CONFIG = {
    "static": {"batch_size": 32, "lr": 5e-4}, # Task 0 + 1
    2: {"batch_size": 32, "lr": 5e-4},        # Task 2
}

MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 15


def load_windowed_data(data_path):
    bundle_path = os.path.join(data_path, "windowed_data_bundle.npz")
    if os.path.exists(bundle_path):
        bundle = np.load(bundle_path, allow_pickle=True)
        windows = bundle["windows"]
        labels = bundle["labels"]
        metadata = pd.DataFrame.from_records(bundle["metadata"])
        return windows, labels, metadata

    windows = np.load(os.path.join(data_path, "windows.npy"))
    labels = np.load(os.path.join(data_path, "labels.npy"))
    metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))
    return windows, labels, metadata


def build_window_features(windows, metadata):
    windows = windows.astype(np.float32)
    task_ids = pd.to_numeric(metadata["taskID"], errors="coerce").fillna(0).astype(int).to_numpy()
    num_task_classes = int(task_ids.max()) + 1 if task_ids.size else 1
    task_one_hot = tf.keras.utils.to_categorical(task_ids, num_classes=num_task_classes).astype(np.float32)
    task_features = np.repeat(task_one_hot[:, None, :], windows.shape[1], axis=1)

    if "isTurn" in metadata.columns:
        is_turn = pd.to_numeric(metadata["isTurn"], errors="coerce").fillna(0).astype(np.float32).to_numpy()
    else:
        is_turn = np.zeros(len(metadata), dtype=np.float32)
    is_turn_features = np.repeat(is_turn[:, None, None], windows.shape[1], axis=1)

    return np.concatenate([windows, task_features, is_turn_features], axis=-1)


def build_subject_bags(windows, labels, metadata):
    subject_keys = metadata.apply(lambda row: (str(row["dataset"]), str(row["subjectID"]).strip()), axis=1)
    bag_map = {}
    for index, key in enumerate(subject_keys):
        bag_map.setdefault(tuple(key), []).append(index)

    bag_windows, bag_labels, bag_keys, bag_sizes = [], [], [], []
    for key, indices in bag_map.items():
        bag_windows.append(windows[indices])
        bag_labels.append(int(pd.Series(labels[indices]).mode().iloc[0]))
        bag_keys.append(key)
        bag_sizes.append(len(indices))

    return bag_windows, np.asarray(bag_labels, dtype=np.int64), bag_keys, np.asarray(bag_sizes, dtype=np.int64)


def pad_bag_list(bag_list):
    max_windows = max(len(bag) for bag in bag_list)
    window_shape = bag_list[0].shape[1:]
    padded = np.zeros((len(bag_list), max_windows, *window_shape), dtype=np.float32)
    mask = np.zeros((len(bag_list), max_windows), dtype=np.float32)

    for bag_index, bag in enumerate(bag_list):
        length = len(bag)
        padded[bag_index, :length] = bag.astype(np.float32)
        mask[bag_index, :length] = 1.0

    return padded, mask

def apply_stretching(X, stretch_factor_range=(0.8, 1.2)):
    batch_size, T, channels = X.shape
    X_stretched = np.empty_like(X)
    
    for i in range(batch_size):
        factor = np.random.uniform(*stretch_factor_range)
        T_new = int(T * factor)
        T_new = max(T_new, 2) 
        
        for c in range(channels):
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, T_new)
            y_new = np.interp(x_new, x_old, X[i, :, c])
            
            X_stretched[i, :, c] = np.interp(np.linspace(0, 1, T), x_new, y_new)
    return X_stretched

def apply_slope(X, max_slope=0.5):
    batch_size, T, channels = X.shape
    X_sloped = X.copy()
    for i in range(batch_size):
        slope = np.linspace(0, np.random.uniform(-max_slope, max_slope), T)
        X_sloped[i] += slope[:, np.newaxis]
    return X_sloped

def apply_flipping(X):
    return X * -1

def augmentation(X, y, target_count=None):
    if target_count is None:
        target_count = np.max(np.bincount(y))
    
    unique_labels = np.unique(y)
    final_X, final_y = [], []

    for label in unique_labels:
        idx = np.where(y == label)[0]
        cls_X = X[idx]
        
        final_X.append(cls_X)
        final_y.append(y[idx])
        
        num_to_add = target_count - len(cls_X)
        
        if num_to_add > 0:
            base_indices = np.random.choice(len(cls_X), num_to_add, replace=True)
            base_samples = cls_X[base_indices].copy()
            technique = np.random.choice(['flipping', 'stretching', 'slope'])
            
            if technique == 'flipping':
                synth = apply_flipping(base_samples)
            elif technique == 'stretching':
                synth = apply_stretching(base_samples)
            elif technique == 'slope':
                synth = apply_slope(base_samples)
                
            final_X.append(synth)
            final_y.append(np.full(num_to_add, label))

    return np.concatenate(final_X), np.concatenate(final_y)

def print_dataset_diagnostics(windows, labels, metadata, bag_labels, bag_sizes):
    print("\n" + "="*50)
    print("PATIENT-LEVEL DIAGNOSTICS")
    print("="*50)
    print(f"Total windows: {len(windows)}")
    print(f"Total patients/bags: {len(bag_labels)}")
    print(f"Global label distribution: {np.bincount(labels)}")
    print(f"Patient label distribution: {np.bincount(bag_labels)}")
    print(f"Window shape: {windows.shape}")
    print(f"Window value range (min/max): [{windows.min():.6f}, {windows.max():.6f}]")
    print(f"Window std: {windows.std():.6f}, mean: {windows.mean():.6f}")
    print(f"Bag size range (min/max): [{bag_sizes.min()}, {bag_sizes.max()}]")
    print(f"Bag size mean: {bag_sizes.mean():.2f}")
    print("="*50 + "\n")


def run_experiment(bag_windows, bag_labels):
    print("\n" + "="*40 + "\nPATIENT-LEVEL CNN-GRU EXPERIMENT\n" + "="*40)

    patient_accuracies = []
    patient_f1s = []
    patient_aucs = []
    config = TASK_CONFIG["static"]
    device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    num_classes = int(np.max(bag_labels)) + 1

    bags_by_class = {}
    for index, label in enumerate(bag_labels):
        bags_by_class.setdefault(int(label), []).append(index)

    train_indices = []
    test_indices = []
    for label, indices in bags_by_class.items():
        sorted_indices = sorted(indices)
        if len(sorted_indices) == 1:
            train_indices.extend(sorted_indices)
            continue

        train_count = max(1, len(sorted_indices) // 2)
        if train_count >= len(sorted_indices):
            train_count = len(sorted_indices) - 1

        train_indices.extend(sorted_indices[:train_count])
        test_indices.extend(sorted_indices[train_count:])

    if not test_indices:
        raise ValueError("The holdout split produced no test subjects. Check the class distribution.")

    train_bags = [bag_windows[index] for index in train_indices]
    test_bags = [bag_windows[index] for index in test_indices]
    y_train = bag_labels[train_indices]
    y_test = bag_labels[test_indices]

    x_train_windows, x_train_mask = pad_bag_list(train_bags)
    x_test_windows, x_test_mask = pad_bag_list(test_bags)
    X_train = {"windows": x_train_windows, "mask": x_train_mask}
    X_test = {"windows": x_test_windows, "mask": x_test_mask}

    print(f"Training patients: {len(train_indices)} | Test patients: {len(test_indices)}")
    print("Patient class distribution before training:", np.bincount(y_train))
    print("Patient class distribution on test:", np.bincount(y_test))

    tf.keras.backend.clear_session()
    model = BagCnnGru(num_classes=num_classes)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = {i: (len(y_train) / (num_classes * count)) if count > 0 else 1.0 for i, count in enumerate(class_counts)}

    model, _, history = fit_model(
        model, X_train, y_train, device_name,
        batch_size=config["batch_size"], max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING_PATIENCE,
        X_val=None, y_val=None, lr=config["lr"], class_weights=class_weights
    )

    y_prob = model.predict(X_test, batch_size=config["batch_size"], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    patient_acc = accuracy_score(y_test, y_pred)
    patient_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    try:
        y_true_bin = label_binarize(y_test, classes=np.arange(num_classes))
        patient_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
    except ValueError:
        patient_auc = float('nan')

    patient_accuracies.append(patient_acc)
    patient_f1s.append(patient_f1)
    patient_aucs.append(patient_auc)

    print(f"Patient Accuracy: {patient_acc:.4f}")
    print(f"Patient F1-macro: {patient_f1:.4f}")
    print(f"Patient AUC-macro: {patient_auc:.4f}")
    print("Patient confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nAverage Patient Accuracy: {np.mean(patient_accuracies):.4f}")
    print(f"Average Patient F1-macro: {np.mean(patient_f1s):.4f}")
    print(f"Average Patient AUC-macro: {np.nanmean(patient_aucs):.4f}")
    return np.mean(patient_accuracies)

def main():
    data_path = "posturalInstability/cnn_gru/data/"
    windows, labels, metadata = load_windowed_data(data_path)

    labels = np.where(labels == 4, 3, labels)
    windows = build_window_features(windows, metadata)
    bag_windows, bag_labels, bag_keys, bag_sizes = build_subject_bags(windows, labels, metadata)

    print_dataset_diagnostics(windows, labels, metadata, bag_labels, bag_sizes)
    result = run_experiment(bag_windows, bag_labels)
    print(f"\nFINAL RESULT: {result:.4f}")

if __name__ == "__main__":
    main()