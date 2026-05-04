import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy.signal import butter, filtfilt, resample_poly

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


matplotlib.rcParams["agg.path.chunksize"] = 10000
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 0.5


SF_DICT = {"fog_star": 60.0, "omnia_park": 90.0, "pd_phone": 200.0, "wearpd": 100.0, "kiel": 200.0}
DATASETS = ["fog_star", "omnia_park", "pd_phone", "wearpd", "kiel"]
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TARGET_HZ = 128
WINDOW_SEC = 5
OVERLAP = 0.2

RAW_DATA_DIR = Path("posturalInstability/data/cleaned_data")
OUTPUT_DIR = Path("posturalInstability/cnn_gru/data/")
REPORT_PATH = Path("posturalInstability/preprocessing_report.pdf")


def plot_comparison(original, processed, ds_name, subject_id, task_id, pdf):
    print(f"Plotting comparison for Dataset: {ds_name}, Subject: {subject_id}, Task: {task_id}")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="col")
    fig.suptitle(f"Dataset: {ds_name} | Subject: {subject_id} | Task: {task_id}", fontsize=16, fontweight="bold")

    titles = ["Original Data (Cleaned)", f"Processed Data ({TARGET_HZ}Hz + Lowpass)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]

    for j in range(2):
        axes[0, j].set_title(titles[j], fontsize=14, pad=15)

    for i, col in enumerate(acc_cols):
        axes[0, 0].plot(original["timestamp"], original[col], label=col, color=colors[i], alpha=0.8, linewidth=1)
        axes[0, 1].plot(processed["timestamp"], processed[col], label=col, color=colors[i], alpha=0.8, linewidth=1, rasterized=True)

    axes[0, 0].set_ylabel("Acceleration [g or m/s²]", fontsize=12)
    axes[0, 0].legend(loc="upper right", fontsize=10)
    axes[0, 1].legend(loc="upper right", fontsize=10)

    for i, col in enumerate(gyro_cols):
        axes[1, 0].plot(original["timestamp"], original[col], label=col, color=colors[i], alpha=0.8, linewidth=1)
        axes[1, 1].plot(processed["timestamp"], processed[col], label=col, color=colors[i], alpha=0.8, linewidth=1, rasterized=True)

    axes[1, 0].set_ylabel("Angular Velocity [rad/s]", fontsize=12)
    axes[1, 0].set_xlabel("Time [s]", fontsize=12)
    axes[1, 1].set_xlabel("Time [s]", fontsize=12)
    axes[1, 0].legend(loc="upper right", fontsize=10)
    axes[1, 1].legend(loc="upper right", fontsize=10)

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


def enforce_nan_policy(group, sf, max_interp_gap_sec=0.2, sensor_cols=None):
    sensor_cols = SENSOR_COLS if sensor_cols is None else sensor_cols
    group = group.sort_values("timestamp").copy()
    max_interp_gap = max(1, int(max_interp_gap_sec * sf))

    for col in sensor_cols:
        isna = group[col].isna().to_numpy()
        if isna.any():
            edges = np.diff(np.r_[False, isna, False].astype(int))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            gap_lengths = ends - starts
            max_gap = int(gap_lengths.max()) if len(gap_lengths) else 0

            if max_gap > max_interp_gap:
                return None

            group[col] = group[col].interpolate(method="linear", limit_direction="both")

        if group[col].isna().any():
            return None

    return group


def trim_outliers(group, sf, z_threshold=2.5, trim_perc=0.15, window_size_sec=3.0):
    group = group.sort_values("timestamp").copy()

    total_duration = group["timestamp"].max() - group["timestamp"].min()
    n_trim = int(trim_perc * total_duration * sf)

    if len(group) < (2 * n_trim + sf):
        return None

    group = group.iloc[n_trim:-n_trim].copy()
    window_size = max(3, int(window_size_sec * sf))

    for col in SENSOR_COLS:
        rolling = group[col].rolling(window=window_size, center=True, min_periods=1)
        z_score = np.abs((group[col] - rolling.mean()) / (rolling.std() + 1e-6))
        group.loc[z_score > z_threshold, col] = np.nan

    group = enforce_nan_policy(group, sf, max_interp_gap_sec=0.2, sensor_cols=SENSOR_COLS)
    if group is None:
        return None

    group["timestamp"] = (group["timestamp"] - group["timestamp"].min()).round(4)
    return group


def apply_lowpass(group, sf, cutoff=20.0):
    nyq = 0.5 * sf
    b, a = butter(4, cutoff / nyq, btype="low")
    for col in SENSOR_COLS:
        centered = group[col].values - group[col].mean()
        group[col] = filtfilt(b, a, centered) + group[col].mean()
    return group


def resample_group(group, original_sf):
    up, down = int(TARGET_HZ), int(original_sf)
    n_target = int(len(group) * TARGET_HZ / original_sf)
    target_idx = np.linspace(0, len(group) - 1, n_target).round().astype(int)

    resampled = {
        "timestamp": np.linspace(0, (len(group) - 1) / original_sf, n_target),
        "subjectID": group["subjectID"].iloc[0],
        "sessionID": group["sessionID"].iloc[0],
        "taskID": int(group["taskID"].iloc[0]),
    }

    for col in SENSOR_COLS:
        padded = np.pad(group[col].values, (int(original_sf), int(original_sf)), mode="reflect")
        signal = resample_poly(padded, up, down)
        resampled[col] = signal[up : up + n_target]

    if "isTurn" in group.columns:
        turn_values = pd.to_numeric(group["isTurn"], errors="coerce").fillna(0).astype(int).to_numpy()
        resampled["isTurn"] = turn_values[target_idx]
    else:
        resampled["isTurn"] = np.zeros(n_target, dtype=int)

    return pd.DataFrame(resampled)


def count_windows_for_group_length(group_length, win_size, overlap):
    if group_length < win_size:
        return 0

    step = max(1, int(round(win_size * (1 - overlap))))
    return 1 + (group_length - win_size) // step


def estimate_label_window_count(df, label, win_size, overlap):
    label_df = df[df["label"] == label]
    total = 0
    for (_, _, _), group in label_df.groupby(["subjectID", "dataset", "taskID"]):
        total += count_windows_for_group_length(len(group), win_size, overlap)
    return total


def choose_label_overlaps(df, target_fraction=0.8, majority_overlap=0.0, candidate_overlaps=None):
    if candidate_overlaps is None:
        candidate_overlaps = np.round(np.linspace(0.1, 0.9, 9), 2)

    win_size = int(TARGET_HZ * WINDOW_SEC)
    labels = sorted(df["label"].dropna().unique())
    base_counts = {label: estimate_label_window_count(df, label, win_size, majority_overlap) for label in labels}

    if not base_counts:
        return {}, 0, None

    majority_label = max(base_counts, key=base_counts.get)
    target_count = max(1, int(np.ceil(base_counts[majority_label] * target_fraction)))

    overlap_by_label = {majority_label: majority_overlap}
    for label in labels:
        if label == majority_label:
            continue

        chosen_overlap = None
        for overlap in candidate_overlaps:
            if overlap <= majority_overlap:
                continue

            if estimate_label_window_count(df, label, win_size, overlap) >= target_count:
                chosen_overlap = float(overlap)
                break

        if chosen_overlap is None:
            chosen_overlap = float(candidate_overlaps[-1])

        overlap_by_label[label] = chosen_overlap

    return overlap_by_label, target_count, majority_label


def create_windows(df, overlap_by_label=None):
    windows, labels, metadata_rows = [], [], []
    win_size = int(TARGET_HZ * WINDOW_SEC)
    window_id = 0

    for label, label_df in df.groupby("label"):
        overlap = OVERLAP if overlap_by_label is None else overlap_by_label.get(label, OVERLAP)
        step = max(1, int(round(win_size * (1 - overlap))))

        for (sid, ds, tid), group in label_df.groupby(["subjectID", "dataset", "taskID"]):
            data = group[SENSOR_COLS].values
            turn_values = pd.to_numeric(group["isTurn"], errors="coerce").fillna(0).astype(int).to_numpy() if "isTurn" in group.columns else np.zeros(len(group), dtype=int)

            if len(data) < win_size:
                continue

            for start_idx in range(0, len(data) - win_size + 1, step):
                end_idx = start_idx + win_size
                windows.append(data[start_idx:end_idx])
                labels.append(int(label))
                metadata_rows.append(
                    {
                        "window_id": window_id,
                        "subjectID": sid,
                        "dataset": ds,
                        "taskID": int(tid),
                        "isTurn": int(turn_values[start_idx:end_idx].max()) if len(turn_values) else 0,
                        "label": int(label),
                        "overlap": float(overlap),
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                    }
                )
                window_id += 1

    return np.asarray(windows), np.asarray(labels, dtype=np.int64), pd.DataFrame(metadata_rows)


def load_processed_sessions():
    processed_sessions = []

    with PdfPages(REPORT_PATH) as pdf:
        for dataset_name in DATASETS:
            sensor_path = RAW_DATA_DIR / f"{dataset_name}_sensor.csv"
            if not sensor_path.exists():
                continue

            df = pd.read_csv(sensor_path)
            required_columns = {"subjectID", "sessionID", "taskID", "timestamp", *SENSOR_COLS}
            missing_columns = required_columns.difference(df.columns)
            if missing_columns:
                raise ValueError(f"{sensor_path} is missing columns: {sorted(missing_columns)}")

            if "isTurn" not in df.columns:
                df["isTurn"] = 0

            df["isTurn"] = pd.to_numeric(df["isTurn"], errors="coerce").fillna(0).astype(int)
            df = df[df["taskID"].isin([0, 1, 2])].copy()

            for (subject_id, session_id, task_id), group in df.groupby(["subjectID", "sessionID", "taskID"]):
                sf = SF_DICT[dataset_name]

                if task_id in [0, 1]:
                    session_group = trim_outliers(group, sf)
                    if session_group is None:
                        continue
                else:
                    session_group = group.sort_values("timestamp").iloc[int(sf):-int(sf)].copy()
                    if len(session_group) <= sf:
                        continue
                    session_group = enforce_nan_policy(session_group, sf, max_interp_gap_sec=0.2, sensor_cols=SENSOR_COLS)
                    if session_group is None:
                        continue

                session_group = enforce_nan_policy(session_group, sf, max_interp_gap_sec=0.2, sensor_cols=SENSOR_COLS)
                if session_group is None:
                    continue

                original_signal = session_group.copy()
                original_signal["timestamp"] = original_signal["timestamp"] - original_signal["timestamp"].iloc[0]

                processed = apply_lowpass(session_group, sf)
                processed = resample_group(processed, sf)
                processed["dataset"] = dataset_name

                # Uncomment if you want a PDF sanity check for the preprocessing.
                # plot_comparison(original_signal, processed, dataset_name, subject_id, task_id, pdf)

                processed_sessions.append(processed)

    if not processed_sessions:
        raise FileNotFoundError("No valid sensor sessions were processed from the cleaned data directory.")

    return pd.concat(processed_sessions, ignore_index=True)


def load_clinical_labels():
    clinical_map = {}
    for dataset_name in DATASETS:
        clinical_path = RAW_DATA_DIR / f"{dataset_name}_clinical.csv"
        if not clinical_path.exists():
            continue

        clinical_df = pd.read_csv(clinical_path, dtype={"subjectID": str})
        if "postural_stability" not in clinical_df.columns:
            continue

        for _, row in clinical_df.iterrows():
            clinical_map[(dataset_name, str(row["subjectID"]).strip())] = row["postural_stability"]

    return clinical_map


def attach_labels(processed_df, clinical_map):
    labeled_df = processed_df.copy()
    labeled_df["label"] = labeled_df.apply(
        lambda row: clinical_map.get((row["dataset"], str(row["subjectID"]).strip()), np.nan),
        axis=1,
    )
    labeled_df = labeled_df.dropna(subset=["label"]).copy()
    labeled_df["label"] = labeled_df["label"].astype(int)
    labeled_df["label"] = np.where(labeled_df["label"] == 4, 3, labeled_df["label"])
    return labeled_df


def save_outputs(x_data, y_data, metadata_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.int64)

    np.save(OUTPUT_DIR / "windows.npy", x_data)
    np.save(OUTPUT_DIR / "labels.npy", y_data)
    metadata_df.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

    np.savez_compressed(
        OUTPUT_DIR / "windowed_data_bundle.npz",
        windows=x_data,
        labels=y_data,
        metadata=metadata_df.to_records(index=False),
        metadata_columns=np.array(metadata_df.columns.to_list(), dtype=object),
    )

    manifest = {
        "target_hz": TARGET_HZ,
        "window_sec": WINDOW_SEC,
        "overlap_default": OVERLAP,
        "sensor_columns": SENSOR_COLS,
        "num_windows": int(len(x_data)),
        "num_classes": int(np.max(y_data) + 1) if len(y_data) else 0,
        "files": {
            "windows": "windows.npy",
            "labels": "labels.npy",
            "metadata": "metadata.csv",
            "bundle": "windowed_data_bundle.npz",
        },
    }

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    processed_df = load_processed_sessions()
    clinical_map = load_clinical_labels()
    labeled_df = attach_labels(processed_df, clinical_map)

    if labeled_df.empty:
        raise ValueError("No labeled sessions remain after joining clinical data.")

    overlap_by_label, target_count, majority_label = choose_label_overlaps(labeled_df, target_fraction=0.5)
    print(f"Majority label: {majority_label} | target windows per minority class: {target_count}")
    print(f"Overlap by label: {overlap_by_label}")

    x_raw, y_final, meta_final = create_windows(labeled_df, overlap_by_label=overlap_by_label)
    if len(x_raw) == 0:
        raise ValueError("No windows were created from the processed sessions.")

    save_outputs(x_raw, y_final, meta_final)

    print(f"Saved {len(x_raw)} windows to {OUTPUT_DIR}")
    print(f"Metadata columns: {list(meta_final.columns)}")


if __name__ == "__main__":
    main()