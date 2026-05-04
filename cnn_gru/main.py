from model import CnnGruModel
from train import train_cnn_lstm_pretrainer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Import moduli locali (Assicurati che i file siano nella stessa cartella)


# Configurazione Percorsi
DATA_PATH = "posturalInstability/cnn_gru/data/"
CHECKPOINT_PATH = "posturalInstability/cnn_gru/models/best_cnn_lstm.keras"
RANDOM_STATE = 42

def load_data():
    """Carica finestre, etichette e metadati."""
    windows = np.load(os.path.join(DATA_PATH, "windows.npy"))
    labels = np.load(os.path.join(DATA_PATH, "labels.npy"))
    metadata = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
    
    # Unifica Classe 4 (Molto Grave) nella Classe 3
    labels = np.where(labels == 4, 3, labels)
    return windows, labels, metadata

def patient_wise_split(metadata, labels):
    """Esegue lo split basato sull'ID del soggetto per evitare data leakage[cite: 4]."""
    subjects = metadata[['dataset', 'subjectID']].drop_duplicates()
    
    # Etichetta prevalente per soggetto per stratificazione
    subj_labels = []
    for _, s in subjects.iterrows():
        mask = (metadata['dataset'] == s['dataset']) & (metadata['subjectID'] == s['subjectID'])
        subj_labels.append(labels[mask][0])
    
    # Split: 60% Train (per pre-training), 40% Test (per valutazione finale)[cite: 4]
    s_train_full, s_test = train_test_split(
        subjects, test_size=0.4, stratify=subj_labels, random_state=RANDOM_STATE
    )
    
    # Sottodivisione Train/Val per il pre-training window-wise
    s_train, s_val = train_test_split(s_train_full, test_size=0.2, random_state=RANDOM_STATE)
    
    return s_train, s_val, s_test

def get_windows_by_subjects(windows, labels, metadata, subjects_subset):
    """Estrae le finestre appartenenti a un gruppo di soggetti[cite: 4]."""
    mask = metadata.apply(lambda r: any((subjects_subset['dataset'] == r['dataset']) & 
                                        (subjects_subset['subjectID'] == r['subjectID'])), axis=1)
    return windows[mask], labels[mask]

def extract_task_aware_features(model, windows, labels, metadata, subjects_subset):
    """
    Estrae Mean, Std, Max, Min dallo spazio latente per ogni Task (0, 1, 2).
    Garantisce che ogni paziente abbia un vettore di lunghezza fissa.
    """
    all_patient_features = []
    patient_labels = []
    expected_tasks = [0, 1, 2]

    for _, s in subjects_subset.iterrows():
        p_mask = (metadata['dataset'] == s['dataset']) & (metadata['subjectID'] == s['subjectID'])
        p_windows = windows[p_mask].astype('float32')
        p_meta = metadata[p_mask]

        # Estrazione embedding latente tramite il modello addestrato[cite: 5]
        z = model.get_latent(p_windows).numpy()
        latent_dim = z.shape[-1]
        task_feature_size = latent_dim * 4  # mean, std, max, min

        patient_vector = []
        for tid in expected_tasks:
            task_indices = (p_meta['taskID'] == tid).values
            if np.any(task_indices):
                z_task = z[task_indices]
                stats = np.concatenate([
                    z_task.mean(axis=0),
                    z_task.std(axis=0),
                    z_task.max(axis=0),
                    z_task.min(axis=0)
                ])
            else:
                # Padding con zeri se il paziente non ha eseguito quel task
                stats = np.zeros(task_feature_size)

            patient_vector.append(stats)

        all_patient_features.append(np.concatenate(patient_vector))
        patient_labels.append(labels[p_mask][0])

    return np.stack(all_patient_features), np.array(patient_labels)

def main():
    # 1. Caricamento Dati[cite: 4]
    windows, labels, metadata = load_data()
    s_train, s_val, s_test = patient_wise_split(metadata, labels)

    # 2. Pre-training Supervisionato Window-wise[cite: 4, 6]
    # Usiamo il modello CNN-LSTM per imparare la dinamica temporale[cite: 5]
    xw_train, yw_train = get_windows_by_subjects(windows, labels, metadata, s_train)
    xw_val, yw_val = get_windows_by_subjects(windows, labels, metadata, s_val)
    
    input_shape = windows.shape[1:]
    num_classes = int(np.max(labels)) + 1
    
    print(f"Inizio Pre-training CNN-LSTM su {len(xw_train)} finestre...")
    model = CnnGruModel(input_shape=input_shape, num_classes=num_classes)
    train_cnn_lstm_pretrainer(model, xw_train, yw_train, xw_val, yw_val, CHECKPOINT_PATH)

    # 3. Estrazione Feature Patient-wise (MIL con Task Context)[cite: 4]
    print("Estrazione feature aggregate per paziente...")
    s_train_full = pd.concat([s_train, s_val])
    X_train_p, y_train_p = extract_task_aware_features(model, windows, labels, metadata, s_train_full)
    X_test_p, y_test_p = extract_task_aware_features(model, windows, labels, metadata, s_test)

    # 4. Benchmarking Modelli Classici[cite: 4]
    clfs = {
        "SVM (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, class_weight='balanced_subsample', random_state=RANDOM_STATE
        )
    }

    for name, clf in clfs.items():
        clf.fit(X_train_p, y_train_p)
        y_pred = clf.predict(X_test_p)
        
        print(f"\n--- Risultati {name} ---")
        print(f"Accuracy: {accuracy_score(y_test_p, y_pred):.4f}")
        print(f"F1-Macro: {f1_score(y_test_p, y_pred, average='macro'):.4f}")
        print("Matrice di Confusione:")
        print(confusion_matrix(y_test_p, y_pred))

        # print t-SNE per visualizzare le feature estratte
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        X_test_tsne = tsne.fit_transform(X_test_p)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_test_p, cmap='viridis', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classi")
        plt.title(f"t-SNE delle feature estratte - {name}")
        plt.xlabel("Dimensione 1")
        plt.ylabel("Dimensione 2")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()