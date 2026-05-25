import tensorflow as tf
from keras.layers import (
    Conv1D, MaxPooling1D, GRU, Dropout, Dense,
    BatchNormalization, GlobalAveragePooling1D,
    Reshape, Conv1DTranspose,
)

@tf.keras.utils.register_keras_serializable()
class CnnGru(tf.keras.Model):

    def __init__(self, input_shape=(640, 6), n_classes=4, latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_spec = input_shape
        self.n_classes  = n_classes
        self.latent_dim = latent_dim

        # Spatial feature extraction
        self.conv1 = Conv1D(32, kernel_size=7, strides=2, activation="relu", padding="same")
        self.bn1   = BatchNormalization()
        self.conv2 = Conv1D(64, kernel_size=5, strides=2, activation="relu", padding="same")
        self.bn2   = BatchNormalization()

        # Temporal modelling
        self.gru     = GRU(64, return_sequences=False)
        self.dropout = Dropout(0.4)

        # Latent representation
        self.dense_feat  = Dense(latent_dim, activation="relu", name="latent_features")

        # Classification head
        self.classifier  = Dense(n_classes, activation="softmax", name="task_output")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.gru(x)
        if training:
            x = self.dropout(x)
        latent = self.dense_feat(x)
        return self.classifier(latent)

    def get_latent(self, inputs, training=False):
        """Extract 16-dim latent vector (no classification head)."""
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.gru(x)
        return self.dense_feat(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2. MASKED AUTOENCODER (self-supervised)
# ──────────────────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable()
class ImuEncoder(tf.keras.Model):
    def __init__(self, input_shape=(640, 6), latent_dim=16, mask_prob=0.3, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_spec = input_shape
        self.latent_dim  = latent_dim
        self.mask_prob   = mask_prob
        self.seq_len     = input_shape[0]          # 640
        self.n_channels  = input_shape[1]          # 6
        self.compressed  = self.seq_len // 4       # 160 (two strides of 2)

        # ── Encoder ────────────────────────────────────────────────────
        self.conv1 = Conv1D(32, kernel_size=7, strides=2, activation="relu", padding="same")
        self.bn1   = BatchNormalization()
        self.conv2 = Conv1D(64, kernel_size=5, strides=2, activation="relu", padding="same")
        self.bn2   = BatchNormalization()

        # GRU processes compressed sequence (seq_len/4 timesteps)
        self.gru        = GRU(64, return_sequences=True)   # keep sequence for decoder
        self.global_avg = GlobalAveragePooling1D()          # for latent extraction
        self.dropout    = Dropout(0.3)

        self.latent_feat = Dense(latent_dim, activation="linear", name="latent_space")

        # ── Decoder ────────────────────────────────────────────────────
        self.dec_dense  = Dense(self.compressed * 64, activation="relu")
        self.dec_reshape = Reshape((self.compressed, 64))
        self.deconv1    = Conv1DTranspose(64, kernel_size=3, strides=2,
                                          activation="relu", padding="same")
        self.deconv2    = Conv1DTranspose(32, kernel_size=3, strides=2,
                                          activation="relu", padding="same")
        self.out_layer  = Conv1D(self.n_channels, kernel_size=3,
                                 activation="linear", padding="same",
                                 name="reconstruction")

    def _mask(self, x):
        mask = tf.cast(
            tf.random.uniform((tf.shape(x)[0], tf.shape(x)[1], 1)) > self.mask_prob,
            tf.float32
        )
        return x * mask

    def _encode(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.gru(x)               
        return x

    def call(self, inputs, training=False):
        x_in  = self._mask(inputs) if training else inputs
        enc   = self._encode(x_in, training=training)

        if training:
            enc = self.dropout(enc)

        # Latent vector from pooled sequence
        pooled = self.global_avg(enc)                  # (batch, 64)
        latent = self.latent_feat(pooled)               # (batch, 16)

        # Decode from latent
        x_dec = self.dec_dense(latent)                 # (batch, compressed*64)
        x_dec = self.dec_reshape(x_dec)               # (batch, compressed, 64)
        x_dec = self.deconv1(x_dec)                   # (batch, seq_len/2, 64)
        x_dec = self.deconv2(x_dec)                   # (batch, seq_len, 32)
        return self.out_layer(x_dec)                  # (batch, seq_len, 6)

    def get_latent(self, inputs, training=False):
        enc    = self._encode(inputs, training=training)
        pooled = self.global_avg(enc)
        return self.latent_feat(pooled)