import os
import tensorflow as tf
import numpy as np


class SparseMacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.f1 = tf.keras.metrics.F1Score(average="macro", threshold=None)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        return self.f1.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.f1.result()

    def reset_state(self):
        self.f1.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

def train_cnn_lstm_pretrainer(model, x_train, y_train, x_val, y_val, checkpoint_path):
    # Abbassiamo drasticamente il learning rate e aggiungiamo il clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

    f1_metric = SparseMacroF1(model.num_classes)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[f1_metric]
    )
    
    # Early stopping più severo per bloccare l'overfitting sul nascere
    weights_path = checkpoint_path if checkpoint_path.endswith(".weights.h5") else checkpoint_path + ".weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(weights_path, monitor="val_f1_score", mode="max", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_f1_score", mode="max", patience=20, restore_best_weights=True)
    ]
    
    return model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=120,
        batch_size=64, # Aumentare il batch size aiuta a stabilizzare la LSTM
        callbacks=callbacks,
        verbose=1
    )