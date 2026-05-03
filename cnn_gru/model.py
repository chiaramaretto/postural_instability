import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout, BatchNormalization, MaxPooling1D, LSTM

def CnnGru(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


layers = tf.keras.layers
models = tf.keras.models


@tf.keras.utils.register_keras_serializable()
class BagCnnGru(models.Model):
    def __init__(self, num_classes, **kwargs):
        super(BagCnnGru, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = 64
        self.conv1 = layers.Conv1D(64, kernel_size=5, activation='relu')
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.conv2 = layers.Conv1D(128, kernel_size=5, activation='relu')
        self.pool2 = layers.MaxPooling1D(pool_size=2)
        self.lstm = layers.LSTM(64, return_sequences=False)
        self.dropout = layers.Dropout(0.3)
        self.dense = layers.Dense(self.embedding_dim, activation='relu')
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def _encode_windows(self, windows, training=False):
        x = self.conv1(windows)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.lstm(x, training=training)
        x = self.dropout(x, training=training)
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
        encoded = self._encode_windows(flat_windows, training=training)
        encoded = tf.reshape(encoded, (batch_size, bag_size, self.embedding_dim))
        encoded.set_shape([None, None, self.embedding_dim])

        if mask is None:
            mask = tf.reduce_any(tf.not_equal(windows, 0.0), axis=[2, 3])

        mask = tf.cast(mask, encoded.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        pooled = tf.reduce_sum(encoded * mask, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
        pooled.set_shape([None, self.embedding_dim])
        return self.classifier(pooled)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)