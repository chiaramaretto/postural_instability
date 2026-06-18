import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import class_weight
from itertools import combinations


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION TRAINING (task recognition)
# ──────────────────────────────────────────────────────────────────────────────

def train_classifier(model, x_train, y_train, x_val, y_val,
          checkpoint_path, weights_filename="best_cnn_gru.weights.h5"):
  
    full_path = os.path.join(checkpoint_path, weights_filename)

    # Balanced class weights
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(enumerate(cw))
    print(f"Training con pesi classe: {cw_dict}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            full_path, monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10,
            min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=64,
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=1,
    )
    plot_classifier_history(history, checkpoint_path)
    return history


def plot_classifier_history(history, save_dir):
    acc     = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss    = history.history["loss"]
    val_loss= history.history["val_loss"]
    epochs  = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, acc,     label="Train")
    ax1.plot(epochs, val_acc, label="Val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, loss,     label="Train")
    ax2.plot(epochs, val_loss, label="Val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "classifier_history.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training history saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# AUTOENCODER TRAINING (self-supervised reconstruction)
# ──────────────────────────────────────────────────────────────────────────────

def train_autoencoder(model, x_train, x_val,
                      checkpoint_path, weights_filename="best_ae.weights.h5"):
   
    full_path = os.path.join(checkpoint_path, weights_filename)

    print(f"Autoencoder training: {x_train.shape[0]} train windows, "
          f"{x_val.shape[0]} val windows")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            full_path, monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10,
            min_lr=1e-6, verbose=1,
        ),
    ]

    # Input = masked signal (handled inside model.call during training)
    # Target = original signal
    return model.fit(
        x_train, x_train,          # target is the original, masking is inside call()
        validation_data=(x_val, x_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

def normalize_latents(latents):
    return tf.nn.l2_normalize(latents, axis=-1)
