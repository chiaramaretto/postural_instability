import os
import tensorflow as tf
import numpy as np


try:
	from .model import Autoencoder
except ImportError:
	from model import Autoencoder


def build_dataset(x, y=None, batch_size=32, shuffle=False):
	if isinstance(x, tf.data.Dataset):
		return x

	if y is None:
		dataset = tf.data.Dataset.from_tensor_slices(x)
	else:
		dataset = tf.data.Dataset.from_tensor_slices((x, y))

	if shuffle:
		dataset = dataset.shuffle(buffer_size=max(len(x), 1))

	return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _subset_inputs(inputs, indices):
	if isinstance(inputs, dict):
		return {key: value[indices] for key, value in inputs.items()}
	return inputs[indices]


def train_autoencoder(
	model,
	x_train,
	y_train,
	x_val=None,
	y_val=None,
	batch_size=32,
	epochs=50,
	learning_rate=1e-3,
	patience=10,
	validation_split=0.2,
	class_weights=None,
	checkpoint_path="best_autoencoder.keras",
):
	if not isinstance(model, tf.keras.Model):
		raise TypeError("model must be a tf.keras.Model instance")

	if x_val is None or y_val is None:
		num_samples = len(y_train)
		validation_size = max(1, int(num_samples * validation_split)) if num_samples > 1 else 0
		train_size = num_samples - validation_size
		if validation_size > 0 and train_size > 0:
			train_indices = np.arange(train_size)
			val_indices = np.arange(train_size, num_samples)
			x_val = _subset_inputs(x_train, val_indices)
			y_val = y_train[val_indices]
			x_train = _subset_inputs(x_train, train_indices)
			y_train = y_train[train_indices]


	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)

	train_dataset = build_dataset(x_train, y_train, batch_size=batch_size, shuffle=True)
	validation_data = None
	if x_val is not None and y_val is not None:
		validation_data = build_dataset(x_val, y_val, batch_size=batch_size, shuffle=False)

	os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_path,
			monitor="val_accuracy" if validation_data is not None else "accuracy",
			save_best_only=True,
			save_weights_only=False,
			verbose=1,
		),
		tf.keras.callbacks.EarlyStopping(
			monitor="val_accuracy" if validation_data is not None else "accuracy",
			patience=patience,
			restore_best_weights=True,
			verbose=1,
		),
	]

	history = model.fit(
		train_dataset,
		validation_data=validation_data,
		epochs=epochs,
		callbacks=callbacks,
		class_weight=class_weights,
		verbose=1,
	)

	if x_val is not None and y_val is not None:
		print("Final confusion matrix:")
		print(tf.math.confusion_matrix(y_val, tf.argmax(model.predict(x_val, verbose=0), axis=1)).numpy())

	return model, history
