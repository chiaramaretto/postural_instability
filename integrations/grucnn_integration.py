"""Script di integrazione: prepara dati, applica preprocessing GRUCNN, esegue CV per ensemble/sostituzione.

Questo script è uno scheletro: riempire `load_data()` per il formato locale dei dati.
"""
from typing import Tuple
import numpy as np

from preprocessing.grucnn_preprocessing import (
    emd_denoise,
    downsample_sequence,
    minmax_scale,
    zero_pad_sequences,
    augment_oversample,
)
from integrations.grucnn_ensemble import GRUCNNWrapper, EnsemblePredictor, prepare_input_for_grucnn


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carica le sequenze e le label dal progetto.

    Deve restituire X: list/array di sequenze (timesteps, channels) e y: labels.
    Implementare secondo la struttura dei dati in `data/`.
    """
    import os
    import pandas as pd

    base = os.path.join("..", "data", "windowed_data")
    axes_dir = os.path.join(base, "axes")

    # Load axis arrays (shape: n_windows x timesteps)
    acc_x = np.load(os.path.join(axes_dir, "acc_x.npy"))
    acc_y = np.load(os.path.join(axes_dir, "acc_y.npy"))
    acc_z = np.load(os.path.join(axes_dir, "acc_z.npy"))
    gyro_x = np.load(os.path.join(axes_dir, "gyro_x.npy"))
    gyro_y = np.load(os.path.join(axes_dir, "gyro_y.npy"))
    gyro_z = np.load(os.path.join(axes_dir, "gyro_z.npy"))

    # Stack channels -> shape (n_samples, timesteps, channels)
    # Some arrays may be (n_timesteps,) if single sample; ensure 2D
    def ensure2d(a):
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        return a

    acc_x = ensure2d(acc_x)
    acc_y = ensure2d(acc_y)
    acc_z = ensure2d(acc_z)
    gyro_x = ensure2d(gyro_x)
    gyro_y = ensure2d(gyro_y)
    gyro_z = ensure2d(gyro_z)

    # check consistent number of samples
    n = acc_x.shape[0]
    for arr in (acc_y, acc_z, gyro_x, gyro_y, gyro_z):
        if arr.shape[0] != n:
            raise ValueError("Inconsistent number of windows across axis files")

    X = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=-1)

    # Load labels from extracted features CSV
    features_csv = os.path.join("..", "data", "extracted_features", "features_clinical.csv")
    df = pd.read_csv(features_csv)

    if "postural_stability" not in df.columns:
        raise ValueError("features_clinical.csv missing 'postural_stability' column")

    y_raw = df["postural_stability"].values

    # Align lengths: drop samples with missing labels
    mask = ~pd.isna(y_raw)
    if mask.sum() != n:
        # filter X and y to available labels
        X = X[mask]
        y = y_raw[mask].astype(float)
    else:
        y = y_raw.astype(float)

    # build bag ids per window using features csv keys
    bag_ids = df["subjectID"].astype(str) + "__" + df["dataset"].astype(str)
    # apply mask if any
    if mask.sum() != n:
        bag_ids = bag_ids[mask].values
    else:
        bag_ids = bag_ids.values

    return X, np.asarray(y), np.asarray(bag_ids)


def preprocess_dataset(X_raw):
    X = []
    for seq in X_raw:
        seq = emd_denoise(seq, n_imf=7)
        seq = downsample_sequence(seq, orig_fs=100, target_fs=25)
        seq = minmax_scale(seq)
        X.append(seq)
    X_pad = zero_pad_sequences(X)
    return X_pad


def run_cv_experiment(X, y, n_splits=5):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math
    import torch
    import torch.nn as nn
    from sklearn.linear_model import Ridge

    # groups must be provided (bag-level ids) — expect caller to pass groups via closure
    raise RuntimeError("run_cv_experiment should be called via run_cv_with_groups(X, y, groups, n_splits)")


def run_cv_with_groups(X, y, groups, n_splits=5):
    """Performs GroupKFold CV where `groups` are bag identifiers per sample.

    Trains three lightweight models per-fold:
      - GRUCNN (PyTorch implementation) on raw windows
      - Attention (uses attention package on bag-structured features)
      - HUF (simple Ridge on extracted features as a stand-in)

    Returns per-fold metrics for each model.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math
    import pandas as pd

    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    # read features CSV for attention/huf bag assembly
    features_csv = ".." + "/data/extracted_features/features_clinical.csv"
    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c.isdigit()]

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        # split arrays
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        # GRUCNN: preprocess and train small PyTorch model on windows
        X_tr_pad = preprocess_dataset(X_train_raw)
        X_te_pad = preprocess_dataset(X_test_raw)

        grucnn_preds_test = _train_and_eval_grucnn_pytorch(X_tr_pad, y_train_raw, groups_train, X_te_pad, groups_test, epochs=8)
        # aggregate to bag-level
        bag_true, bag_pred = _aggregate_by_bag(groups_test, y_test_raw, grucnn_preds_test)
        grucnn_mae = float(mean_absolute_error(bag_true, bag_pred))
        grucnn_rmse = float(math.sqrt(mean_squared_error(bag_true, bag_pred)))

        # Attention: build bags from features_df and train using attention training utilities
        attention_metrics = _train_attention_on_groups(df, feature_cols, groups, train_idx, test_idx)

        # HUF (light): train Ridge on per-window features and average per-bag
        huf_pred_test = _train_simple_huf(df, feature_cols, groups, train_idx, test_idx)
        bag_true_h, bag_pred_h = _aggregate_by_bag(groups[test_idx], y[test_idx], huf_pred_test)
        huf_mae = float(mean_absolute_error(bag_true_h, bag_pred_h))
        huf_rmse = float(math.sqrt(mean_squared_error(bag_true_h, bag_pred_h)))

        fold_results.append({
            "fold": fold,
            "grucnn": {"mae": grucnn_mae, "rmse": grucnn_rmse},
            "attention": attention_metrics,
            "huf": {"mae": huf_mae, "rmse": huf_rmse},
        })

    return fold_results


def _aggregate_by_bag(groups_arr, y_arr, preds_arr):
    # groups_arr: bag id per sample (same length as y_arr and preds)
    import numpy as _np
    unique_bags = []
    bag_true = []
    bag_pred = []
    seen = set()
    for g, yv, pv in zip(groups_arr, y_arr, preds_arr):
        if g not in seen:
            # collect all indices with this bag
            idxs = _np.where(groups_arr == g)[0]
            bag_true.append(float(_np.mean(y_arr[idxs])))
            bag_pred.append(float(_np.mean(preds_arr[idxs])))
            seen.add(g)
    return _np.array(bag_true), _np.array(bag_pred)


def _train_and_eval_grucnn_pytorch(X_tr, y_tr, groups_tr, X_te, groups_te, epochs=8):
    """Train a small PyTorch GRU+2CNN and return per-window predictions on X_te."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # X_* shape: (n, timesteps, channels)
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float().unsqueeze(1)
    X_te_t = torch.from_numpy(X_te).float()

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    class SimpleGRUCNN(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.cnn1 = nn.Sequential(
                nn.Conv1d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.cnn2 = nn.Sequential(
                nn.Conv1d(channels, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.gru = nn.GRU(input_size=channels, hidden_size=64, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(64 + 32 + 64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

        def forward(self, x):
            # x: (batch, timesteps, channels)
            b, t, c = x.shape
            x_c = x.permute(0, 2, 1)  # (batch, channels, timesteps)
            f1 = self.cnn1(x_c).view(b, -1)
            f2 = self.cnn2(x_c).view(b, -1)
            _, h = self.gru(x)
            h = h[-1]
            cat = torch.cat([f1, f2, h], dim=1)
            return self.fc(cat).squeeze(1)

    model = SimpleGRUCNN(channels=X_tr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred.unsqueeze(1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"GRUCNN epoch {epoch+1}/{epochs} loss={running/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_te_t.to(device)).cpu().numpy()

    return preds


def _train_attention_on_groups(features_df, feature_cols, groups_all, train_idx, test_idx):
    """Train attention model using bag-structured features from features_df for the provided split.

    Returns metrics dict from evaluate().
    """
    import pandas as pd
    import torch
    from attention.main_att import SubjectBagDataset, collate_subject_bags
    from attention.model import SubjectTaskAttentionModel
    from attention.train import fit_model, evaluate
    from torch.utils.data import DataLoader

    # Build subject bags from features_df
    features_df_local = features_df.copy()
    features_df_local['subjectID'] = features_df_local['subjectID'].astype(str)
    features_df_local['dataset'] = features_df_local['dataset'].astype(str)

    # Build bag list
    subject_bags = []
    for (subj_id, dataset), group in features_df_local.groupby(['subjectID', 'dataset']):
        y_val = pd.to_numeric(group['postural_stability'].iloc[0], errors='coerce')
        if pd.isna(y_val):
            continue
        subject_bags.append({
            'subjectID': subj_id,
            'dataset': dataset,
            'bag_id': f"{subj_id}__{dataset}",
            'x': group[feature_cols].values,
            'task_ids': group['taskID'].values,
            'y': float(y_val),
        })

    # Map bag indices to sample indices using groups_all
    # Determine which bag_ids belong to train/test by checking groups_all indices
    train_mask = np.zeros(len(groups_all), dtype=bool)
    train_mask[train_idx] = True
    test_mask = np.zeros(len(groups_all), dtype=bool)
    test_mask[test_idx] = True

    # Determine bag ids sets
    train_bag_set = set(groups_all[train_mask])
    test_bag_set = set(groups_all[test_mask])

    train_bags = [b for b in subject_bags if b['bag_id'] in train_bag_set]
    test_bags = [b for b in subject_bags if b['bag_id'] in test_bag_set]

    if not train_bags or not test_bags:
        print("Warning: no bags for attention in this fold; skipping attention training")
        return {"mae": float('nan'), "rmse": float('nan'), "acc_rounded": float('nan')}

    train_ds = SubjectBagDataset(train_bags)
    val_ds = SubjectBagDataset(test_bags)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_subject_bags)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_subject_bags)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SubjectTaskAttentionModel(input_dim=len(feature_cols), hidden_dim=128, num_tasks=3, dropout=0.3).to(device)

    model = fit_model(model, train_loader, val_loader, device, max_epochs=20, patience=6)
    metrics = evaluate(model, val_loader, device)
    return metrics


def _train_simple_huf(features_df, feature_cols, groups_all, train_idx, test_idx):
    """Train a Ridge regressor on per-window features and return per-window predictions for test set."""
    from sklearn.linear_model import Ridge
    import numpy as _np
    import pandas as pd

    X_feat = features_df[feature_cols].values.astype(float)
    y_feat = pd.to_numeric(features_df['postural_stability'], errors='coerce').values.astype(float)

    # mask out NaNs
    valid = ~_np.isnan(y_feat)
    X_feat = X_feat[valid]
    y_feat = y_feat[valid]
    groups_valid = groups_all[valid]

    train_mask = _np.zeros(len(groups_all), dtype=bool)
    train_mask[train_idx] = True
    test_mask = _np.zeros(len(groups_all), dtype=bool)
    test_mask[test_idx] = True

    # Filter to valid
    train_mask = train_mask[valid]
    test_mask = test_mask[valid]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_feat[train_mask], y_feat[train_mask])
    preds = ridge.predict(X_feat[test_mask])
    # return preds aligned to test_mask order
    return preds


if __name__ == "__main__":
    X_raw, y, groups = load_data()
    print(f"Loaded data: X={X_raw.shape}, y={y.shape}, groups={len(groups)}")
    results = run_cv_with_groups(X_raw, y, groups, n_splits=5)
    print("\nCV Results:")
    for r in results:
        print(r)
