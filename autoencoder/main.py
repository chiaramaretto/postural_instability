from model import Autoencoder
from train import train_window_pretrainer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


# Parametri Globali
DATA_PATH = "posturalInstability/cnn_gru/data/"
CHECKPOINT = "posturalInstability/autoencoder/checkpoints/best_pretrain.keras"
RANDOM_STATE = 42

def load_data():
    windows = np.load(os.path.join(DATA_PATH, "windows.npy"))
    labels = np.load(os.path.join(DATA_PATH, "labels.npy"))
    metadata = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
    labels = np.where(labels == 4, 3, labels) # Unifica le classi di instabilità grave
    return windows, labels, metadata

def patient_wise_split(metadata, labels):
    """Divide i soggetti garantendo l'indipendenza dei set."""
    subjects = metadata[['dataset', 'subjectID']].drop_duplicates()
    subj_labels = [labels[(metadata['dataset']==s['dataset']) & (metadata['subjectID']==s['subjectID'])][0] 
                   for _, s in subjects.iterrows()]
    
    s_train_full, s_test = train_test_split(subjects, test_size=0.4, stratify=subj_labels, random_state=RANDOM_STATE)
    s_train, s_val = train_test_split(s_train_full, test_size=0.2, random_state=RANDOM_STATE)
    return s_train, s_val, s_test

def get_windows_for_pretrain(windows, labels, metadata, subjects_subset):
    mask = metadata.apply(lambda r: any((subjects_subset['dataset'] == r['dataset']) & 
                                        (subjects_subset['subjectID'] == r['subjectID'])), axis=1)
    return windows[mask], labels[mask]

def extract_task_aware_features(model, windows, labels, metadata, subjects_subset):
    all_patient_features = []
    patient_labels = []
    
    # Definiamo i task attesi (0, 1, 2)
    expected_tasks = [0, 1, 2]
    embedding_dim = 8# Deve corrispondere a self.embedding_dim nel model.py
    num_stats = 4 # mean, std, max, min
    
    for _, s in subjects_subset.iterrows():
        p_mask = (metadata['dataset'] == s['dataset']) & (metadata['subjectID'] == s['subjectID'])
        p_metadata = metadata[p_mask]
        p_windows = windows[p_mask].astype('float32')
        
        # Estrazione embedding (Batch processing per velocità)
        z = model.get_latent(p_windows).numpy()
        
        patient_vector = []
        for tid in expected_tasks:
            # Maschera booleana per il task specifico del paziente
            task_indices = (p_metadata['taskID'] == tid).values
            
            if np.any(task_indices):
                z_task = z[task_indices]
                stats = np.concatenate([
                    z_task.mean(axis=0),
                    z_task.std(axis=0),
                    z_task.max(axis=0),
                    z_task.min(axis=0)
                ])
            else:
                # Se il task manca, riempiamo con zeri della stessa dimensione
                stats = np.zeros(embedding_dim * num_stats)
            
            patient_vector.append(stats)
            
        # Uniamo i vettori dei 3 task in un unico "super-vettore"
        all_patient_features.append(np.concatenate(patient_vector))
        patient_labels.append(labels[p_mask][0])
        
    return np.array(all_patient_features), np.array(patient_labels)

def main():
    windows, labels, metadata = load_data()
    s_train, s_val, s_test = patient_wise_split(metadata, labels)

    # 1. Pre-training Window-wise (Solo su pazienti di training)
    xw_train, yw_train = get_windows_for_pretrain(windows, labels, metadata, s_train)
    xw_val, yw_val = get_windows_for_pretrain(windows, labels, metadata, s_val)
    
    model = Autoencoder(num_classes=len(np.unique(labels)))
    train_window_pretrainer(model, xw_train, yw_train, xw_val, yw_val, CHECKPOINT)

    # 2. Feature Extraction Patient-wise (Task-Aware)
    X_train, y_train = extract_task_aware_features(model, windows, labels, metadata, pd.concat([s_train, s_val]))
    X_test, y_test = extract_task_aware_features(model, windows, labels, metadata, s_test)

    # 3. Benchmark Modelli
    clfs = {
        "SVM": Pipeline([('s', StandardScaler()), ('m', SVC(kernel='rbf', class_weight='balanced', probability=True))]),
        "Random Forest": RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, objective='multi:softprob', random_state=RANDOM_STATE)
    }

    for name, clf in clfs.items():
        if name == "XGBoost":
            clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        else:
            clf.fit(X_train, y_train)
            
        y_pred = clf.predict(X_test)
        print(f"\n--- {name} ---")
        print(f"F1 macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(confusion_matrix(y_test, y_pred))

    # 4. t-SNE
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE).fit_transform(X_test)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=y_test, cmap='viridis', edgecolors='k')
    plt.title("t-SNE Task-Aware Patient Features")
    plt.show()

if __name__ == "__main__":
    main()