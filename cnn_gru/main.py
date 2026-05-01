import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from model import CnnGru
from train import fit_model, predict, print_confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


def confusion_matrix(y_true, y_pred, datasets, class_labels):
    datasets = np.asarray(datasets)
    unique_datasets = pd.Series(datasets).dropna().unique()

    print("\nConfusion matrix:")
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

    # Build subject keys (dataset, subjectID) so the same subject doesn't appear across splits
    subject_keys = metadata.apply(lambda r: (r['dataset'], str(r['subjectID']).strip()), axis=1)
    metadata = metadata.reset_index(drop=True)

    # Map each unique subject key to the indices of its windows (preserving order)
    subj_to_idx = {}
    for idx, key in enumerate(subject_keys):
        subj_to_idx.setdefault(key, []).append(idx)

    unique_subjects = list(subj_to_idx.keys())

    # For stratification at subject-level, compute the majority label per subject
    subj_labels = []
    for key in unique_subjects:
        idxs = subj_to_idx[key]
        vals = labels[idxs]
        # use the mode; if tie, take first
        vals_nonzero = vals
        if len(vals_nonzero) == 0:
            subj_labels.append(0)
        else:
            subj_labels.append(int(pd.Series(vals_nonzero).mode().iloc[0]))

    # Split subjects into train+temp and test. Use stratification if every subject-level class
    # has at least 2 members; otherwise fall back to a non-stratified split.
    subj_label_counts = pd.Series(subj_labels).value_counts()
    if (subj_label_counts < 2).any() or len(unique_subjects) < 2:
        print("Warning: some subject-level classes have <2 members; using non-stratified subject split.")
        train_subj, test_subj = train_test_split(
            unique_subjects,
            test_size=0.2,
            random_state=42,
        )
    else:
        train_subj, test_subj = train_test_split(
            unique_subjects,
            test_size=0.2,
            random_state=42,
            stratify=subj_labels
        )

    # Split train_subj into train and val (val ~= 0.1 overall)
    # compute stratify labels for train_subj and fallback if needed
    train_subj_labels = [subj_labels[unique_subjects.index(s)] for s in train_subj]
    train_label_counts = pd.Series(train_subj_labels).value_counts()
    if (train_label_counts < 2).any() or len(train_subj) < 2:
        print("Warning: some labels within the training subjects have <2 members; using non-stratified train/val subject split.")
        train_subj_final, val_subj = train_test_split(
            train_subj,
            test_size=0.125,
            random_state=42,
        )
    else:
        train_subj_final, val_subj = train_test_split(
            train_subj,
            test_size=0.125,  # 0.125 of 0.8 ~= 0.1 overall
            random_state=42,
            stratify=train_subj_labels
        )

    # Build index lists
    train_idx = [i for s in train_subj_final for i in subj_to_idx[s]]
    val_idx = [i for s in val_subj for i in subj_to_idx[s]]
    test_idx = [i for s in test_subj for i in subj_to_idx[s]]

    # Preserve original ordering of windows within each split
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)

    X_train, y_train, meta_train = windows[train_idx], labels[train_idx], metadata.iloc[train_idx].reset_index(drop=True)
    X_val, y_val, meta_val = windows[val_idx], labels[val_idx], metadata.iloc[val_idx].reset_index(drop=True)
    X_test, y_test, meta_test = windows[test_idx], labels[test_idx], metadata.iloc[test_idx].reset_index(drop=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnGru(input_channels=6).to(device)
    # Define augmentation / balancing function (used only on training set)
    def oversample_dataset_subset(windows_arr, labels_arr, metadata_df, k_neighbors=3):
        labels_arr = np.asarray(labels_arr)
        unique_labels, counts = np.unique(labels_arr, return_counts=True)
        if len(unique_labels) == 0:
            return windows_arr, labels_arr, metadata_df

        target_count = int(counts.max())
        balanced_windows = []
        balanced_labels = []
        balanced_metadata = []

        for label in unique_labels:
            idx = np.where(labels_arr == label)[0]
            cls_windows = windows_arr[idx]
            cls_metadata = metadata_df.iloc[idx].reset_index(drop=True)

            if len(cls_windows) >= target_count:
                chosen_idx = np.random.default_rng(42).choice(len(cls_windows), size=target_count, replace=False)
                balanced_windows.append(cls_windows[chosen_idx])
                balanced_labels.append(np.full(target_count, label, dtype=np.int64))
                balanced_metadata.append(cls_metadata.iloc[chosen_idx].reset_index(drop=True))
                continue

            flat_windows = cls_windows.reshape(len(cls_windows), -1)
            nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(cls_windows)), metric="euclidean")
            nn.fit(flat_windows)
            knns = nn.kneighbors(flat_windows, return_distance=False)

            synth_windows = []
            synth_metadata = []
            num_to_add = target_count - len(cls_windows)
            rng = np.random.default_rng(42)

            for _ in range(num_to_add):
                i = rng.integers(0, len(cls_windows))
                neighbor_candidates = knns[i][1:]
                if len(neighbor_candidates) == 0:
                    neighbor_idx = i
                else:
                    neighbor_idx = rng.choice(neighbor_candidates)

                alpha = rng.random()
                synthetic_sample = cls_windows[i] + alpha * (cls_windows[neighbor_idx] - cls_windows[i])
                noise = rng.normal(0, 0.001, synthetic_sample.shape)
                synth_windows.append(synthetic_sample + noise)
                synth_metadata.append(cls_metadata.iloc[i].to_dict())

            balanced_windows.append(np.concatenate([cls_windows, np.stack(synth_windows)]))
            balanced_labels.append(np.full(target_count, label, dtype=np.int64))
            balanced_metadata.append(pd.concat([cls_metadata, pd.DataFrame(synth_metadata)], ignore_index=True))

        return (
            np.concatenate(balanced_windows),
            np.concatenate(balanced_labels),
            pd.concat(balanced_metadata, ignore_index=True),
        )

    # Apply dataset-wise balancing/augmentation only on the training set
    aug_windows = []
    aug_labels = []
    aug_meta = []
    for ds in meta_train['dataset'].unique():
        mask = meta_train['dataset'] == ds
        if not mask.any():
            continue
        ds_w = X_train[mask.to_numpy()]
        ds_y = y_train[mask.to_numpy()]
        ds_meta = meta_train.loc[mask].reset_index(drop=True)

        bw, by, bm = oversample_dataset_subset(ds_w, ds_y, ds_meta, k_neighbors=3)
        aug_windows.append(bw)
        aug_labels.append(by)
        aug_meta.append(bm)

    if aug_windows:
        X_train = np.concatenate(aug_windows, axis=0)
        y_train = np.concatenate(aug_labels, axis=0)
        meta_train = pd.concat(aug_meta, ignore_index=True)

    # Shuffle training set after augmentation
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    meta_train = meta_train.iloc[perm].reset_index(drop=True)

    # 4. Fit (provide explicit validation set)
    m, acc, history = fit_model(model, X_train, y_train, device, batch_size=32, max_epochs=500, patience=20, X_val=X_val, y_val=y_val)
    print(f"Best validation accuracy: {acc:.4f}")

    # Final test evaluation
    # Test predictions and probabilities for AUC
    y_pred = predict(m, X_test, device, batch_size=32)

    # Compute probabilities for AUC
    m.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(X_test, dtype=torch.float32).to(device)
        outputs = m(Xt)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    labels_unique = np.unique(labels)

    confusion_matrix(y_test, y_pred, meta_test["dataset"].to_numpy(), labels_unique)

    # Metrics
    acc_test = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    # Specificity: compute per-class and macro-average
    cm = confusion_matrix(y_test, y_pred, labels=labels_unique)
    spec_per_class = []
    for i in range(len(labels_unique)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        spec_per_class.append(spec)
    specificity = float(np.mean(spec_per_class))

    # AUC (macro)
    try:
        auc = roc_auc_score(np.eye(np.max(y_test) + 1)[y_test], probs, average='macro', multi_class='ovo')
    except Exception:
        auc = 0.0

    print(f"Test metrics - Acc: {acc_test:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall (sens): {recall:.4f}, Specificity: {specificity:.4f}, AUC: {auc:.4f}")

    # Save and plot training/validation loss
    os.makedirs('posturalInstability/cnn_gru/data/plots', exist_ok=True)
    try:
        plt.figure()
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train/Validation Loss')
        plt.savefig('posturalInstability/cnn_gru/data/plots/loss.png')
        plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()