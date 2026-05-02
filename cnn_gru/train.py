import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix
def fit_model(
    model,
    X_train,
    y_train,
    device_name, 
    batch_size=16,
    max_epochs=100,
    val_fraction=0.2,
    patience=20,
    X_val=None,
    y_val=None,
    class_weights=None,
    lr=1e-4,
):
    os.makedirs('posturalInstability/cnn_gru/data/models', exist_ok=True)
    model_path = 'posturalInstability/cnn_gru/data/models/best_stacking_model.h5'

    # 1. Preparazione Pesi Classi (TF richiede un dizionario {class_index: weight})
    cw_dict = None
    if class_weights is not None:
        cw_dict = {i: w for i, w in enumerate(class_weights)}

    # 2. Compilazione del Modello
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. Callbacks (Early Stopping e salvataggio miglior modello)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=patience, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]

    # 4. Addestramento
    # Usiamo il contesto del dispositivo specificato
    with tf.device(device_name):
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val)
        else:
            val_data = None # TF gestirà val_fraction se X_val è None

        history_obj = model.fit(
            X_train, y_train,
            validation_data=val_data,
            validation_split=val_fraction if val_data is None else 0.0,
            epochs=max_epochs,
            batch_size=batch_size,
            class_weight=cw_dict,
            callbacks=callbacks,
            verbose=1
        )

    # 5. Estrazione History per compatibilità con il tuo codice precedente
    history = {
        'train_loss': history_obj.history['loss'],
        'val_loss': history_obj.history['val_loss'],
        'val_accuracy': history_obj.history['val_accuracy'],
    }

    # Nota: Keras non calcola F1/AUC per ogni epoca di default nel modo esatto di sklearn, 
    # ma i pesi migliori sono già stati ripristinati dall'EarlyStopping.
    
    return model, max(history['val_accuracy']), history


def predict(model, X, device_name, batch_size=32):
    with tf.device(device_name):
        probs = model.predict(X, batch_size=batch_size, verbose=0)
        predictions = np.argmax(probs, axis=1)
    return predictions


def print_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Final confusion matrix:")
    print(cm)
    return cm