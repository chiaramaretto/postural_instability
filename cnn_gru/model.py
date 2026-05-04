import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, GlobalAveragePooling1D

@tf.keras.utils.register_keras_serializable()
class CnnGruModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes, **kwargs):
        super(CnnGruModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.input_shape_spec = input_shape
        
        # Estrazione Feature Spaziali
        self.conv1 = Conv1D(31, kernel_size=5, activation='relu', padding='same')
        self.pool1 = MaxPooling1D(pool_size=2)
        self.conv2 = Conv1D(64, kernel_size=5, activation='relu', padding='same')
        self.pool2 = MaxPooling1D(pool_size=2)
        
        # Analisi Temporale (Sostituisce la logica statica dell'Autoencoder)[cite: 5]
        self.lstm = LSTM(32, return_sequences=False)
        self.dropout = Dropout(0.3)
        
        self.dense_feat = Dense(32, activation='relu', name="latent_features")
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.lstm(x)
        if training:
            x = self.dropout(x)
        latent = self.dense_feat(x)
        return self.classifier(latent)

    def get_latent(self, inputs):
        """Estrae l'embedding latente per la classificazione Patient-wise."""
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.lstm(x)
        return self.dense_feat(x)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "input_shape": self.input_shape_spec})
        return config