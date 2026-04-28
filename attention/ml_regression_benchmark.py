import torch
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, Dataset

try:
    from imblearn.over_sampling import ADASYN, SMOTE
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

from model import SubjectTaskAttentionModel
from train import evaluate as evaluate_attention
from train import fit_model


@dataclass
class FoldMetrics:
    mae: float
    rmse: float


@dataclass
class AttentionConfig:
    name: str
    use_class_weights: bool = False


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
    y = torch.zeros((len(batch),), dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        x[i, :n] = torch.from_numpy(np.asarray(item["x"], dtype=np.float32))
        task_ids[i, :n] = torch.from_numpy(np.asarray(item["task_ids"], dtype=np.int64))
        window_mask[i, :n] = True
        y[i] = float(item["y"])

    return {
        "x": x,
        "task_ids": task_ids,
        "window_mask": window_mask,
        "y": y,
    }


def resolve_features_path(repo_root: str) -> str:
    path = os.path.join(repo_root, "data", "extracted_features", "features_clinical.csv")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"Features file not found at {path}"
    )

def build_subject_level_table(features_df: pd.DataFrame) -> pd.DataFrame:
    features_df = features_df.copy()
    features_df["subjectID"] = features_df["subjectID"].astype(str)
    features_df["dataset"] = features_df["dataset"].astype(str)

    feature_cols = [c for c in features_df.columns if c.isdigit()]
    group_cols = ["subjectID", "dataset"]

    if not feature_cols:
        raise ValueError("No numeric feature columns found (expected digit-named columns).")
    if "postural_stability" not in features_df.columns:
        raise ValueError("Column 'postural_stability' not found.")

    subj_features = features_df.groupby(group_cols)[feature_cols].mean()
    subj_target = features_df.groupby(group_cols)["postural_stability"].first()

    subject_df = pd.concat([subj_features, subj_target], axis=1).reset_index()
    subject_df["postural_stability"] = pd.to_numeric(subject_df["postural_stability"], errors="coerce")
    subject_df = subject_df[subject_df["postural_stability"].notna()].copy()
    subject_df = subject_df[(subject_df["postural_stability"] >= 0) & (subject_df["postural_stability"] <= 4)].copy()

    return subject_df


def build_subject_bags(features_df: pd.DataFrame):
    feature_cols = [c for c in features_df.columns if c.isdigit()]
    if "taskID" not in features_df.columns:
        raise ValueError("Column 'taskID' not found. Needed for Attention model.")

    subject_bags = []
    for (subj_id, dataset), group in features_df.groupby(["subjectID", "dataset"]):
        y_val = pd.to_numeric(group["postural_stability"].iloc[0], errors="coerce")
        if pd.isna(y_val):
            continue
        if y_val < 0 or y_val > 4:
            continue

        subject_bags.append(
            {
                "subjectID": str(subj_id),
                "dataset": str(dataset),
                "x": group[feature_cols].to_numpy(),
                "task_ids": pd.to_numeric(group["taskID"], errors="coerce").fillna(0).astype(int).to_numpy(),
                "y": float(y_val),
            }
        )

    if not subject_bags:
        raise ValueError("No valid subject bags found for Attention benchmark.")

    return subject_bags


def compute_target_class_weights(train_bags, n_classes=5):
    y_classes = np.clip(np.round([b["y"] for b in train_bags]).astype(int), 0, n_classes - 1)
    counts = np.bincount(y_classes, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()
    return weights


def augment_rare_bags(train_bags, class_weights, repeat_factor=1):
    aug_bags = list(train_bags)
    rng = np.random.default_rng(42)

    for bag in train_bags:
        y_cls = int(np.clip(np.round(bag["y"]), 0, len(class_weights) - 1))
        if class_weights[y_cls] <= 1.2:
            continue

        n = bag["x"].shape[0]
        if n < 2:
            continue

        for rep in range(repeat_factor):
            idx = rng.choice(n, size=n, replace=True)
            aug_bags.append(
                {
                    "subjectID": bag["subjectID"],
                    "dataset": bag["dataset"],
                    "x": bag["x"][idx],
                    "task_ids": bag["task_ids"][idx],
                    "y": bag["y"],
                }
            )

    return aug_bags


def _augment_attention_bags_with_resampler(train_bags, method, random_state=42):
    """
    Apply SMOTE/ADASYN at window level inside the TRAIN fold only.
    Each window inherits the bag ordinal class (rounded target 0-4),
    synthetic windows are appended back into real bags of the same class.
    """
    if not HAS_IMBLEARN:
        return train_bags

    X_rows = []
    y_rows = []
    max_task_id = 0
    for bag in train_bags:
        y_cls = int(np.clip(np.round(bag["y"]), 0, 4))
        x = np.asarray(bag["x"], dtype=np.float32)
        tids = np.asarray(bag["task_ids"], dtype=np.int64)
        if x.shape[0] == 0:
            continue
        max_task_id = max(max_task_id, int(tids.max()) if tids.size else 0)

        rows = np.concatenate([x, tids.reshape(-1, 1).astype(np.float32)], axis=1)
        X_rows.append(rows)
        y_rows.append(np.full(rows.shape[0], y_cls, dtype=np.int32))

    if not X_rows:
        return train_bags

    Xw = np.vstack(X_rows)
    yw = np.concatenate(y_rows)
    sampling = _safe_sampling_dict(yw)
    if not sampling:
        return train_bags

    if method == "smote":
        sampler = SMOTE(sampling_strategy=sampling, random_state=random_state, k_neighbors=1)
    elif method == "adasyn":
        sampler = ADASYN(sampling_strategy=sampling, random_state=random_state, n_neighbors=1)
    else:
        return train_bags

    try:
        X_res, y_res = sampler.fit_resample(Xw, yw)
    except Exception:
        return train_bags

    n_orig = Xw.shape[0]
    if X_res.shape[0] <= n_orig:
        return train_bags

    X_syn = X_res[n_orig:]
    y_syn = y_res[n_orig:]

    class_to_bag_indices = {}
    for i, bag in enumerate(train_bags):
        c = int(np.clip(np.round(bag["y"]), 0, 4))
        class_to_bag_indices.setdefault(c, []).append(i)

    rng = np.random.default_rng(random_state)
    add_x = [[] for _ in train_bags]
    add_t = [[] for _ in train_bags]

    for row, c in zip(X_syn, y_syn):
        bag_candidates = class_to_bag_indices.get(int(c), [])
        if not bag_candidates:
            continue
        bag_i = int(rng.choice(bag_candidates))
        feat = row[:-1].astype(np.float32)
        tid = int(np.clip(np.round(row[-1]), 0, max_task_id))
        add_x[bag_i].append(feat)
        add_t[bag_i].append(tid)

    aug_bags = []
    for i, bag in enumerate(train_bags):
        x = np.asarray(bag["x"], dtype=np.float32)
        t = np.asarray(bag["task_ids"], dtype=np.int64)
        if add_x[i]:
            x_aug = np.vstack([x, np.asarray(add_x[i], dtype=np.float32)])
            t_aug = np.concatenate([t, np.asarray(add_t[i], dtype=np.int64)])
        else:
            x_aug, t_aug = x, t

        aug_bags.append(
            {
                "subjectID": bag["subjectID"],
                "dataset": bag["dataset"],
                "x": x_aug,
                "task_ids": t_aug,
                "y": bag["y"],
            }
        )

    return aug_bags


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5):
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("Need at least 2 unique groups for grouped CV.")

    n_splits = min(n_splits, unique_groups)
    if n_splits < 2:
        n_splits = 2

    splitter = GroupKFold(n_splits=n_splits)
    fold_results = []

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0, 4)

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        fold_results.append(FoldMetrics(mae=mae, rmse=rmse))

    maes = np.array([m.mae for m in fold_results], dtype=np.float32)
    rmses = np.array([m.rmse for m in fold_results], dtype=np.float32)

    return {
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=0)),
        "n_folds": len(fold_results),
    }


def _safe_sampling_dict(y_cls):
    classes, counts = np.unique(y_cls, return_counts=True)
    max_count = int(counts.max())
    sampling = {}
    for c, cnt in zip(classes, counts):
        # Need at least 2 samples in class to run neighbor-based synthetic over-sampling.
        if int(cnt) >= 2 and int(cnt) < max_count:
            sampling[int(c)] = max_count
    return sampling


def _resample_train_ordinal(X_train, y_train, method):
    if not HAS_IMBLEARN:
        return X_train, y_train, False

    y_cls = np.clip(np.round(y_train.to_numpy()).astype(int), 0, 4)
    sampling = _safe_sampling_dict(y_cls)
    if not sampling:
        return X_train, y_train, False

    if method == "smote":
        sampler = SMOTE(sampling_strategy=sampling, random_state=42, k_neighbors=1)
    elif method == "adasyn":
        sampler = ADASYN(sampling_strategy=sampling, random_state=42, n_neighbors=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_res, y_res_cls = sampler.fit_resample(X_train.to_numpy(), y_cls)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res_cls.astype(np.float32), name=y_train.name)
    return X_res, y_res, True


def evaluate_model_with_resampling(model, X, y, groups, method, n_splits=5):
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("Need at least 2 unique groups for grouped CV.")

    n_splits = min(n_splits, unique_groups)
    if n_splits < 2:
        n_splits = 2

    splitter = GroupKFold(n_splits=n_splits)
    fold_results = []
    used_any_resampling = False

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        try:
            X_train_fit, y_train_fit, used = _resample_train_ordinal(X_train, y_train, method=method)
            used_any_resampling = used_any_resampling or used
        except Exception:
            # If synthetic oversampling is not feasible for this fold, fallback to original train fold.
            X_train_fit, y_train_fit = X_train, y_train

        model.fit(X_train_fit, y_train_fit)
        preds = np.clip(model.predict(X_test), 0, 4)

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        fold_results.append(FoldMetrics(mae=mae, rmse=rmse))

    maes = np.array([m.mae for m in fold_results], dtype=np.float32)
    rmses = np.array([m.rmse for m in fold_results], dtype=np.float32)
    return {
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=0)),
        "n_folds": len(fold_results),
        "used_resampling": used_any_resampling,
    }


def evaluate_attention_model(subject_bags, config: AttentionConfig, n_splits: int = 5):
    groups = np.array([b["subjectID"] for b in subject_bags])
    y = np.array([b["y"] for b in subject_bags], dtype=np.float32)

    unique_groups = len(np.unique(groups))
    if unique_groups < 2:
        raise ValueError("Need at least 2 unique groups for Attention grouped CV.")

    n_splits = min(n_splits, unique_groups)
    if n_splits < 2:
        n_splits = 2

    splitter = GroupKFold(n_splits=n_splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(subject_bags)), y, groups=groups), start=1):
        torch.manual_seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)

        train_bags = [subject_bags[i] for i in train_idx]
        test_bags = [subject_bags[i] for i in test_idx]

        train_ds = SubjectBagDataset(train_bags)
        test_ds = SubjectBagDataset(test_bags)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_subject_bags)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_subject_bags)

        input_dim = train_bags[0]["x"].shape[1]
        max_task_id = max(int(np.max(b["task_ids"])) for b in subject_bags)

        model = SubjectTaskAttentionModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_tasks=max_task_id + 1,
            dropout=0.3,
        ).to(device)

        model = fit_model(
            model,
            train_loader,
            test_loader,
            device,
            max_epochs=80,
            patience=12,
            min_delta=1e-3,
            scheduler_patience=5,
            scheduler_factor=0.5,
            min_lr=1e-5,
        )

        metrics = evaluate_attention(model, test_loader, device)
        fold_results.append(FoldMetrics(mae=float(metrics["mae"]), rmse=float(metrics["rmse"])))

    maes = np.array([m.mae for m in fold_results], dtype=np.float32)
    rmses = np.array([m.rmse for m in fold_results], dtype=np.float32)

    return {
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=0)),
        "n_folds": len(fold_results),
    }


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = resolve_features_path(repo_root)

    print("=" * 80)
    print("ML REGRESSION BENCHMARK (GROUPED CV)")
    print("=" * 80)
    print(f"Features file: {features_path}")

    features_df = pd.read_csv(features_path, dtype={"subjectID": str, "dataset": str}, low_memory=False)
    features_df["subjectID"] = features_df["subjectID"].astype(str)
    features_df["dataset"] = features_df["dataset"].astype(str)

    subject_df = build_subject_level_table(features_df)
    subject_bags = build_subject_bags(features_df)

    feature_cols = [c for c in subject_df.columns if c.isdigit()]
    X = subject_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = subject_df["postural_stability"].astype(float)

    # Group by subjectID to avoid leakage when same subject appears in multiple datasets.
    groups = subject_df["subjectID"].astype(str)

    print(f"Subject-dataset samples: {len(subject_df)}")
    print(f"Unique subjects (groups): {groups.nunique()}")
    print(f"Attention bags: {len(subject_bags)}")
    print("Target distribution:")
    print(y.value_counts().sort_index().to_string())
    print("-" * 80)

    models = {
        "Dummy(mean)": DummyRegressor(strategy="mean"),
        "ElasticNet": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=42, max_iter=10000)),
            ]
        ),
        "SVR(RBF)": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVR(C=3.0, epsilon=0.15, gamma="scale")),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGBR": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            random_state=42,
        ),
    }

    rows = []
    for name, model in models.items():
        scores = evaluate_model(model, X, y, groups, n_splits=5)
        rows.append(
            {
                "model": name,
                "mae_mean": scores["mae_mean"],
                "mae_std": scores["mae_std"],
                "rmse_mean": scores["rmse_mean"],
                "rmse_std": scores["rmse_std"],
                "folds": scores["n_folds"],
            }
        )

    if HAS_IMBLEARN:
        resampling_models = {
            "RandomForest_SMOTE": (RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ), "smote"),
            "RandomForest_ADASYN": (RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ), "adasyn"),
            "ElasticNet_SMOTE": (Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=42, max_iter=10000)),
                ]
            ), "smote"),
            "ElasticNet_ADASYN": (Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=42, max_iter=10000)),
                ]
            ), "adasyn"),
        }

        print("Evaluating ML models with fold-wise SMOTE/ADASYN...")
        for name, (model, method) in resampling_models.items():
            scores = evaluate_model_with_resampling(model, X, y, groups, method=method, n_splits=5)
            rows.append(
                {
                    "model": name,
                    "mae_mean": scores["mae_mean"],
                    "mae_std": scores["mae_std"],
                    "rmse_mean": scores["rmse_mean"],
                    "rmse_std": scores["rmse_std"],
                    "folds": scores["n_folds"],
                }
            )
    else:
        print("imblearn not available: skipping SMOTE/ADASYN experiments.")

    attention_configs = [
        AttentionConfig(name="Attention_base", use_class_weights=False),
    ]

    print("Training and evaluating Attention ablations with grouped CV...")
    for cfg in attention_configs:
        print(f"  -> {cfg.name}")
        att_scores = evaluate_attention_model(subject_bags, cfg, n_splits=5)
        rows.append(
            {
                "model": cfg.name,
                "mae_mean": att_scores["mae_mean"],
                "mae_std": att_scores["mae_std"],
                "rmse_mean": att_scores["rmse_mean"],
                "rmse_std": att_scores["rmse_std"],
                "folds": att_scores["n_folds"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values(by="mae_mean", ascending=True).reset_index(drop=True)

    print("Results (sorted by MAE mean):")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out_path = os.path.join(repo_root, "attention", "ml_regression_benchmark_results.csv")
    results_df.to_csv(out_path, index=False)
    print("-" * 80)
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
