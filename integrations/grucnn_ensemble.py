"""Stub per integrazione/ensemble di GRUCNN con modelli esistenti.

Questo file contiene wrapper e funzioni helper per:
- descrivere l'architettura GRUCNN (da completare leggendo GRUCNN.pdf)
- fornire un'interfaccia compatibile con gli attuali `model.py` / `model_att.py`
- effettuare una semplice aggregazione/ensemble delle predizioni

TODO: estrarre i dettagli architetturali da `GRUCNN.pdf` e implementare `build_grucnn`.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import Model
except Exception:
    tf = None  # type: ignore


class GRUCNNWrapper:
    """Wrapper placeholder per il modello GRUCNN.

    Implementare `build_model` seguendo GRUCNN.pdf e adattare `preprocess_input` ai
    dati della pipeline esistente.
    """

    def __init__(self, input_shape: Sequence[int], num_classes: int = 2):
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes
        self.model: Optional[Model] = None

    def build_model(self) -> Model:
        """Costruisce l'architettura GRUCNN come stacking ensemble interno:

        - due teste 1D-CNN (shallow)
        - una testa GRU (shallow)
        - concatenazione delle rappresentazioni e meta-learner (dense)

        Questa architettura si basa sulla descrizione del paper (2x1D-CNN + 1xGRU,
        stacking ensemble con meta-learner) e fornisce un'implementazione riproducibile.
        """
        if tf is None:
            raise RuntimeError("TensorFlow non disponibile nell'ambiente")

        inputs = tf.keras.Input(shape=self.input_shape)

        # CNN head A
        x1 = tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
        x1 = tf.keras.layers.MaxPooling1D(pool_size=2)(x1)
        x1 = tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x1)
        x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)

        # CNN head B (different receptive field)
        x2 = tf.keras.layers.Conv1D(16, kernel_size=5, activation="relu", padding="same")(inputs)
        x2 = tf.keras.layers.MaxPooling1D(pool_size=2)(x2)
        x2 = tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu", padding="same")(x2)
        x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)

        # GRU head
        x3 = tf.keras.layers.GRU(64, return_sequences=False)(inputs)

        # Concatenate heads and meta-learner
        x = tf.keras.layers.Concatenate()([x1, x2, x3])
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="GRUCNN_2CNN_1GRU")
        return self.model

    def load_weights(self, path: str) -> None:
        if self.model is None:
            self.build_model()
        self.model.load_weights(path)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.build_model()
        return self.model.predict(x)


class EnsemblePredictor:
    """Semplice ensemble che combina predizioni da più modelli.

    Esempio d'uso:
      - costruire wrapper per `attention` e `huf` che espongono `predict(x)`
      - costruire `GRUCNNWrapper` e passarli a `EnsemblePredictor`
    """

    def __init__(self, models: Sequence[Any], weights: Optional[Sequence[float]] = None):
        self.models = list(models)
        if weights is None:
            self.weights = [1.0 for _ in self.models]
        else:
            self.weights = list(weights)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Aggrega le probabilità pesate dei modelli.

        Assumiamo che ogni modello esponga `predict(x)` che ritorna
        probabilità di shape (n_samples, n_classes).
        """
        probs = None
        total_weight = sum(self.weights)
        for model, weight in zip(self.models, self.weights):
            model_probs = model.predict(x)
            if probs is None:
                probs = weight * model_probs
            else:
                probs += weight * model_probs

        if probs is None:
            raise RuntimeError("Nessun modello fornito all'EnsemblePredictor")

        probs = probs / total_weight
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)


def prepare_input_for_grucnn(raw_input: np.ndarray) -> np.ndarray:
    """Placeholder preprocessing per adattare i dati alla GRUCNN.

    Sostituire con i passi reali: normalizzazione, windowing, reshape, ecc.
    """
    # Assume already shaped (n_samples, timesteps, channels)
    return raw_input.astype(np.float32)


if __name__ == "__main__":
    # Esempio minimale: crea il wrapper e stampa il summary
    import os

    example_shape = (128, 3)
    wrapper = GRUCNNWrapper(input_shape=example_shape, num_classes=2)
    try:
        model = wrapper.build_model()
        model.summary()
    except RuntimeError:
        print("TensorFlow non disponibile; questo è solo uno stub.")
