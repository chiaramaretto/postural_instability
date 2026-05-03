import tensorflow as tf


layers = tf.keras.layers
models = tf.keras.models


@tf.keras.utils.register_keras_serializable()
class Autoencoder(models.Model):
    def __init__(self, num_classes, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.num_classes = num_classes
        # Encoder 
        self.conv1 = layers.Conv1D(32, 3, activation="relu", padding="same")
        self.pool1 = layers.MaxPooling1D(2, padding="same")
        self.conv2 = layers.Conv1D(64, 3, activation="relu", padding="same")
        self.pool2 = layers.MaxPooling1D(2, padding="same")
        self.conv3 = layers.Conv1D(128, 3, activation="relu", padding="same")
        # Latent space
        self.latent_space = layers.MaxPooling1D(2, padding="same", name="latent_space")
         # classification head
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(64, activation="relu")
        self.classification_output = layers.Dense(num_classes, activation="softmax", name="classification_output")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        encoded = self.latent_space(x)
        x = self.flatten(encoded)
        x = self.dense(x)
        return self.classification_output(x)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class BagAutoencoder(models.Model):
    def __init__(self, num_classes, **kwargs):
        super(BagAutoencoder, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = 64
        self.conv1 = layers.Conv1D(32, 3, activation="relu", padding="same")
        self.pool1 = layers.MaxPooling1D(2, padding="same")
        self.conv2 = layers.Conv1D(64, 3, activation="relu", padding="same")
        self.pool2 = layers.MaxPooling1D(2, padding="same")
        self.conv3 = layers.Conv1D(128, 3, activation="relu", padding="same")
        self.latent_space = layers.MaxPooling1D(2, padding="same", name="latent_space")
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(self.embedding_dim, activation="relu")
        self.classification_output = layers.Dense(num_classes, activation="softmax", name="classification_output")

    def _encode_windows(self, windows):
        x = self.conv1(windows)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.latent_space(x)
        x = self.global_pool(x)
        return self.dense(x)

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            windows = inputs["windows"]
            mask = inputs.get("mask")
        elif isinstance(inputs, (tuple, list)):
            windows, mask = inputs
        else:
            windows = inputs
            mask = None

        windows_shape = tf.shape(windows)
        batch_size = windows_shape[0]
        bag_size = windows_shape[1]
        time_steps = windows_shape[2]
        channels = windows_shape[3]

        flat_windows = tf.reshape(windows, (-1, time_steps, channels))
        encoded = self._encode_windows(flat_windows)
        encoded = tf.reshape(encoded, (batch_size, bag_size, self.embedding_dim))
        encoded.set_shape([None, None, self.embedding_dim])

        if mask is None:
            mask = tf.reduce_any(tf.not_equal(windows, 0.0), axis=[2, 3])

        mask = tf.cast(mask, encoded.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        summed = tf.reduce_sum(encoded * mask, axis=1)
        counts = tf.reduce_sum(mask, axis=1)
        pooled = summed / tf.maximum(counts, 1.0)
        pooled.set_shape([None, self.embedding_dim])
        return self.classification_output(pooled)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


