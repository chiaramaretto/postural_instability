import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

@tf.keras.utils.register_keras_serializable()
class Autoencoder(models.Model):
    def __init__(self, num_classes, **kwargs):
        # Rimuoviamo num_classes dagli argomenti passati a super()
        super(Autoencoder, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = 8
        
        # Encoder
        self.conv1 = layers.Conv1D(16, 3, activation="relu", padding="same")
        self.pool1 = layers.MaxPooling1D(2, padding="same")
        self.latent_space = layers.GlobalAveragePooling1D(name="latent_space")
        
        self.dense = layers.Dense(self.embedding_dim, activation="relu")
        self.out = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        latent = self.latent_space(x)
        x = self.dense(latent)
        return self.out(x)

    def get_latent(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        latent = self.latent_space(x)
        return self.dense(latent)

    # METODI NECESSARI PER LA SERIALIZZAZIONE (Risolvono l'errore)
    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({
            "num_classes": self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)