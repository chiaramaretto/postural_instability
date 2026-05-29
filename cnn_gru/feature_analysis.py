import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/"
RESULTS_PATH = "posturalInstability/cnn_gru/results/"
FEATURE_PREFIX = "train_features_enriched_"
HANDCRAFTED_FEATURE_NAMES = [
    "sway_area_mean", "lat_dom_mean", "p_sway_mean", "p_tremor_mean",
    "sway_area_std", "lat_dom_std", "p_sway_std", "p_tremor_std",
    "jerk_mean", "step_cv_mean", "dom_freq_mean",
    "jerk_std", "step_cv_std", "dom_freq_std",
]
DEFAULT_BOX_FEATURES = ["sway_area_mean", "step_cv_mean", "dom_freq_mean"]


def discover_feature_mode(checkpoint_path=CHECKPOINT_PATH, preferred_modes=None):
    preferred_modes = preferred_modes or ["classifier_mmd", "classifier", "autoencoder"]
    available_modes = []

    for fname in os.listdir(checkpoint_path):
        if not fname.startswith(FEATURE_PREFIX) or not fname.endswith(".csv"):
            continue

        mode = fname[len(FEATURE_PREFIX):-4]
        train_path = os.path.join(checkpoint_path, f"train_features_enriched_{mode}.csv")
        test_path = os.path.join(checkpoint_path, f"test_features_enriched_{mode}.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            available_modes.append(mode)

    if not available_modes:
        raise FileNotFoundError("No enriched feature CSV pairs found in the models folder.")

    for preferred in preferred_modes:
        for mode in available_modes:
            if mode == preferred or mode.startswith(preferred):
                return mode

    return sorted(available_modes)[0]


def load_feature_frame(mode):
    train_path = os.path.join(CHECKPOINT_PATH, f"train_features_enriched_{mode}.csv")
    test_path = os.path.join(CHECKPOINT_PATH, f"test_features_enriched_{mode}.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing enriched feature files for mode '{mode}'.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df["split"] = "train"
    test_df["split"] = "test"
    return pd.concat([train_df, test_df], ignore_index=True)


def get_feature_columns(df):
    feat_cols = [c for c in df.columns if c.startswith("Feat_")]
    if len(feat_cols) <= len(HANDCRAFTED_FEATURE_NAMES):
        raise ValueError("Expected latent features followed by handcrafted features, but the frame is too small.")
    return feat_cols


def split_latent_and_handcrafted(df):
    feat_cols = get_feature_columns(df)
    latent_cols = feat_cols[:-len(HANDCRAFTED_FEATURE_NAMES)]
    handcrafted_cols = feat_cols[-len(HANDCRAFTED_FEATURE_NAMES):]

    latent_df = df[latent_cols].copy()
    handcrafted_df = df[handcrafted_cols].copy()
    handcrafted_df.columns = HANDCRAFTED_FEATURE_NAMES
    handcrafted_df["dataset"] = df["dataset"].values
    handcrafted_df["label"] = df["y_true"].values

    return latent_df, handcrafted_df


def _lazy_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except ImportError as first_error:
        try:
            import umap.umap_ as umap  # type: ignore
            return umap
        except ImportError:
            raise ImportError(
                "UMAP is not installed. Install `umap-learn` in the active environment to run this analysis."
            ) from first_error


def make_umap_plot(df, output_dir, mode):
    umap = _lazy_import_umap()
    latent_df, _ = split_latent_and_handcrafted(df)
    latent_values = latent_df.to_numpy(dtype=np.float32)

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for dataset_name in pd.unique(df["dataset"]):
        mask = df["dataset"] == dataset_name
        axes[0].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=dataset_name,
            alpha=0.65,
            s=35,
        )
    axes[0].set_title("Latent space - colore per dataset")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend(loc="best", fontsize=8)

    label_colors = {0: "steelblue", 1: "tomato"}
    label_names = {0: "HC", 1: "PD"}
    for lbl, col in label_colors.items():
        mask = df["y_true"] == lbl
        axes[1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=col,
            label=label_names[lbl],
            alpha=0.65,
            s=35,
        )
    axes[1].set_title("Latent space - colore per classe")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].legend(loc="best")

    fig.suptitle(f"UMAP latent space - {mode}", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"umap_latent_{mode}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_handcrafted_boxplots(df, output_dir, mode, features=None):
    _, handcrafted_df = split_latent_and_handcrafted(df)
    features = features or DEFAULT_BOX_FEATURES

    sns.set_theme(style="whitegrid")
    output_paths = []

    for feature in features:
        if feature not in handcrafted_df.columns:
            raise ValueError(f"Feature '{feature}' is not available in the handcrafted feature frame.")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(data=handcrafted_df, x="dataset", y=feature, hue="label", ax=ax)
        ax.set_title(f"{feature} - {mode}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(feature)
        ax.legend(title="Label", labels=["HC", "PD"])
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"boxplot_{feature}_{mode}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP and handcrafted feature analyses for CNN-GRU enriched features.")
    parser.add_argument("--mode", default="auto", help="Feature mode to load, or 'auto' to discover the preferred pair.")
    parser.add_argument(
        "--analysis",
        default="all",
        choices=["all", "umap", "handcrafted"],
        help="Which analysis to run.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Handcrafted features to plot. Defaults to sway_area_mean, step_cv_mean, and dom_freq_mean.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = discover_feature_mode() if args.mode == "auto" else args.mode
    df = load_feature_frame(mode)

    output_dir = os.path.join(RESULTS_PATH, f"feature_analysis_{mode}")
    os.makedirs(output_dir, exist_ok=True)

    generated = []
    if args.analysis in ("all", "umap"):
        generated.append(make_umap_plot(df, output_dir, mode))
    if args.analysis in ("all", "handcrafted"):
        generated.extend(make_handcrafted_boxplots(df, output_dir, mode, args.features))

    print(f"Saved analysis outputs to {output_dir}")
    for path in generated:
        print(f"  -> {path}")


if __name__ == "__main__":
    main()