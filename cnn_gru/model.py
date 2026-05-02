import tensorflow as tf
from tensorflow.keras import layers, models

def CnnGru(input_shape=(640, 6), num_classes=4):

    inputs = tf.keras.Input(shape=input_shape)

    c1 = layers.Conv1D(32, kernel_size=1, activation='relu')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.MaxPooling1D(pool_size=2)(c1)
    c1 = layers.GlobalAveragePooling1D()(c1) 

    c2 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.MaxPooling1D(pool_size=2)(c2)
    c2 = layers.GlobalAveragePooling1D()(c2)

    g = layers.GRU(16, return_sequences=True)(inputs)
    g = layers.TimeDistributed(layers.Dense(16))(g)
    g = layers.GlobalAveragePooling1D()(g)

    combined = layers.concatenate([c1, c2, g])
    x = layers.Dropout(0.5)(combined)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model