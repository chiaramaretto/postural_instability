import os
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

try:
	from .model import BagAutoencoder
	from .train import train_autoencoder
except ImportError:
	from model import BagAutoencoder
	from train import train_autoencoder


warnings.filterwarnings("ignore")


DATA_PATH = os.getenv(
	"AUTOENCODER_DATA_PATH",
	"posturalInstability/cnn_gru/data/",
)
CHECKPOINT_DIR = os.getenv(
	"AUTOENCODER_CHECKPOINT_DIR",
	"posturalInstability/autoencoder/checkpoints",
)
N_SPLITS = 5
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 20


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


def build_model_inputs(windows, metadata):
	windows = windows.astype(np.float32)
	task_ids = pd.to_numeric(metadata["taskID"], errors="coerce").fillna(0).astype(int).to_numpy()
	num_task_classes = int(task_ids.max()) + 1 if task_ids.size else 1
	task_one_hot = tf.keras.utils.to_categorical(task_ids, num_classes=num_task_classes).astype(np.float32)
	task_features = np.repeat(task_one_hot[:, None, :], windows.shape[1], axis=1)

	if "isTurn" in metadata.columns:
		is_turn = pd.to_numeric(metadata["isTurn"], errors="coerce").fillna(0).astype(np.float32).to_numpy()
	else:
		print("WARNING: metadata.csv does not contain 'isTurn'. Using zeros for this feature.")
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


def print_dataset_diagnostics(windows, labels, metadata, bag_labels, bag_sizes):
	print("\n" + "=" * 50)
	print("DATASET DIAGNOSTICS")
	print("=" * 50)
	print(f"Total windows: {len(windows)}")
	print(f"Total patients/bags: {len(bag_labels)}")
	print(f"Global label distribution: {np.bincount(labels)}")
	print(f"Patient label distribution: {np.bincount(bag_labels)}")
	print(f"Window shape: {windows.shape}")
	print(f"Window value range (min/max): [{windows.min():.6f}, {windows.max():.6f}]")
	print(f"Window std: {windows.std():.6f}, mean: {windows.mean():.6f}")
	print(f"Bag size range (min/max): [{bag_sizes.min()}, {bag_sizes.max()}]")
	print(f"Bag size mean: {bag_sizes.mean():.2f}")
	if "taskID" in metadata.columns:
		print(f"Task IDs present: {sorted(metadata['taskID'].dropna().unique().tolist())}")
	if "isTurn" in metadata.columns:
		print(f"isTurn distribution: {np.bincount(pd.to_numeric(metadata['isTurn'], errors='coerce').fillna(0).astype(int))}")

	if "taskID" in metadata.columns:
		for task_id in sorted(metadata["taskID"].dropna().unique()):
			task_mask = metadata["taskID"] == task_id
			if task_mask.any():
				print(f"Task {task_id} label distribution: {np.bincount(labels[task_mask])}")

	print("=" * 50 + "\n")


def run_experiment(bag_windows, bag_labels, bag_keys):
	print("\n" + "=" * 40)
	print("PATIENT-LEVEL AUTOENCODER EXPERIMENT")
	print("=" * 40)

	device_name = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
	num_classes = int(np.max(bag_labels)) + 1
	bags_by_class = {}
	for index, label in enumerate(bag_labels):
		bags_by_class.setdefault(int(label), []).append(index)

	train_indices = []
	test_indices = []
	for label, indices in bags_by_class.items():
		sorted_indices = sorted(indices, key=lambda idx: bag_keys[idx])
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
	x_train = {"windows": x_train_windows, "mask": x_train_mask}
	x_test = {"windows": x_test_windows, "mask": x_test_mask}

	print(f"Training patients: {len(train_indices)} | Test patients: {len(test_indices)}")
	print("Patient class distribution before training:", np.bincount(y_train))
	print("Patient class distribution on test:", np.bincount(y_test))

	tf.keras.backend.clear_session()
	model = BagAutoencoder(num_classes=num_classes)
	checkpoint_path = os.path.join(CHECKPOINT_DIR, "autoencoder_holdout.keras")
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)

	with tf.device(device_name):
		model, history = train_autoencoder(
			model,
			x_train,
			y_train,
			x_val=None,
			y_val=None,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			patience=PATIENCE,
			validation_split=0.2,
			checkpoint_path=checkpoint_path,
		)

	y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
	y_pred = np.argmax(y_prob, axis=1)
	patient_accuracy = accuracy_score(y_test, y_pred)
	patient_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
	try:
		y_true_bin = label_binarize(y_test, classes=np.arange(num_classes))
		patient_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
	except ValueError:
		patient_auc = float("nan")

	print(f"Patient Accuracy: {patient_accuracy:.4f}")
	print(f"Patient F1-macro: {patient_f1:.4f}")
	print(f"Patient AUC-macro: {patient_auc:.4f}")
	print("Patient confusion matrix:")
	print(confusion_matrix(y_test, y_pred))
	print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
	return patient_accuracy


def main():
	windows, labels, metadata = load_windowed_data(DATA_PATH)

	labels = np.where(labels == 4, 3, labels)

	windows = build_model_inputs(windows, metadata)
	bag_windows, bag_labels, bag_keys, bag_sizes = build_subject_bags(windows, labels, metadata)

	print_dataset_diagnostics(windows, labels, metadata, bag_labels, bag_sizes)

	result = run_experiment(bag_windows, bag_labels, bag_keys)
	print(f"\nFINAL RESULT: {result:.4f}")


if __name__ == "__main__":
	main()
