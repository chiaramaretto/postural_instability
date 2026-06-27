# Postural Instability Detection in Parkinson's Disease Across Multiple Datasets

A cross-dataset deep learning pipeline for detecting postural instability in Parkinson's Disease (PD) from a single lower-back inertial measurement unit (IMU). The system trains a shared encoder on windowed IMU signals, extracts patient-level features combining deep latent representations with handcrafted biomechanical descriptors, and applies domain adaptation to mitigate distributional shift across heterogeneous datasets.

---

## Overview

Postural instability is one of the most disabling motor symptoms of Parkinson's Disease and a key predictor of falls. Assessing it objectively and automatically from wearable sensors is clinically valuable, but existing studies typically train and test on a single dataset, limiting generalisability. This work addresses the cross-dataset problem: five publicly available datasets are harmonised into a common format, a CNN-GRU encoder is trained to extract a compact latent representation, and lightweight classifiers operating at the patient level are evaluated with and without domain adaptation.

**Key contributions:**
- Adaptive windowing strategy that balances class distributions across heterogeneous recording conditions
- Two encoder variants: a self-supervised masked autoencoder (`ImuEncoder`) and a supervised CNN-GRU classifier (`CnnGru`)
- Patient-level feature set combining deep latent features with task-specific handcrafted biomechanical descriptors
- Evaluation of CORAL and MMD mean-shift domain adaptation on cross-dataset postural instability classification
- Parallel domain separability analysis to quantify how much domain shift is reduced by each adaptation method

---

## Datasets

The pipeline uses five datasets, each recorded with an IMU sensor worn on the lower back:

| Dataset     | Sampling rate | Subjects |
|-------------|---------------|----------|
| FOG-STAR    | 60 Hz         | -        |
| Omnia-Park  | 90 Hz         | -        |
| PD-Phone    | 200 Hz        | -        |
| WearPD      | 100 Hz        | -        |
| KIEL        | 200 Hz        | -        |

Each dataset is expected in two CSV files per dataset: `<name>_sensor.csv` (raw IMU signals) and `<name>_clinical.csv` (demographic and clinical data including the postural stability label). The data is not included in this repository due to redistribution restrictions. Please refer to the original dataset publications to obtain access.

The required common format after harmonisation (produced by `clean_data.ipynb`) is:

- **Sensor CSV columns**: `subjectID`, `taskID`, `sessionID`, `dataset`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`
- **Clinical CSV columns**: `subjectID`, `postural_stability`, `age`, `gender`, `disease_duration`
- Axes orientation: vertical (x), antero-posterior (y), medio-lateral (z)
- Task codes: `0` / `1` = Stance, `2` = Walking

The postural stability label is merged into four ordinal classes: 0 (normal), 1 (mild impairment), 2 (moderate), 3 (severe — originally ≥ 3 on the clinical scale).

---

## Repository Structure

```
posturalInstability/
├── data/                        # not tracked — populate locally
│   ├── cleaned_data/            # output of clean_data.ipynb
│   └── preprocessed_data/       # output of preprocessing.py
│       ├── windows.npy          # (N, 320, 6) float32
│       ├── labels.npy           # (N,) float32
│       └── metadata.csv         # window-level metadata
├── models/                      # not tracked — created at runtime
├── results/                     # not tracked — created at runtime
├── extract_turn/                # turn detection module (Pham algorithm)
├── clean_data.ipynb             # step 1: dataset harmonisation
├── preprocessing.py             # step 2: filtering, resampling, windowing
├── model.py                     # CNN-GRU and masked autoencoder definitions
├── train.py                     # training loops for encoder and classifier
├── feature_extraction.py        # step 3: encoder training + feature extraction + domain adaptation
├── classifier.py                # step 4a: patient-level postural instability classification
├── domain_classifier.py         # step 4b: domain separability analysis
├── save_latent.py               # utility: latent trend analysis per subject
├── explore_datasets.ipynb       # window-level exploratory analysis
├── results_analysis.ipynb       # visualisation and statistical analysis of results
└── run_pipeline.py              # orchestrates steps 2–4 end-to-end
```

---

## Pipeline

The pipeline has four sequential steps. Steps 2 through 4 can be run individually or through `run_pipeline.py`.

### Step 1 — Dataset harmonisation (`clean_data.ipynb`)

Manually curated notebook that reads the raw files from each dataset and writes standardised sensor and clinical CSVs into `data/cleaned_data/`. This step is dataset-specific and requires access to the original data files.

### Step 2 — Preprocessing (`preprocessing.py`)

Reads from `data/cleaned_data/` and writes to `data/preprocessed_data/`.

- **Filtering**: Butterworth bandpass filter (0.5–20 Hz, order 4) applied to walking segments; lowpass filter (20 Hz) applied to stance segments
- **Outlier removal**: rolling Z-score (window = 2 s); samples with |Z| > 5 are set to NaN
- **Resampling**: all datasets resampled to 64 Hz using polyphase resampling (`scipy.signal.resample_poly`)
- **Edge trimming**: first and last 5 % of each recording discarded to remove transient artefacts
- **Subject filtering**: subjects without both a walking and a stance segment, or without a clinical label, are excluded
- **Adaptive windowing**: 5-second windows (320 samples at 64 Hz); overlap per class is automatically computed to bring each minority class to 85 % of the majority count, within the range [0.50, 0.85]; windows with more than 5 % NaN are discarded; remaining NaNs are interpolated with a degree-3 polynomial
- **Output**: `windows.npy` (N × 320 × 6), `labels.npy` (N,), `metadata.csv`

```bash
python preprocessing.py
```

### Step 3 — Encoder training and feature extraction (`feature_extraction.py`)

Reads preprocessed windows and trains (or loads) the shared encoder, then extracts patient-level features and applies domain adaptation. Accepts `--arch-mode` (`autoencoder` or `classifier`) and `--seed`.

**Encoder architectures** (defined in `model.py`):

- `CnnGru` (supervised): Conv1D(16, k=5) → BN → Conv1D(32, k=3) → BN → GRU(32) → Dense(8, relu) → Dense(n\_classes, softmax). The 8-dimensional output of the penultimate layer is used as the latent representation.
- `ImuEncoder` (self-supervised masked autoencoder): same encoder path with a GRU producing a sequence, followed by GlobalAveragePooling and Dense(8, linear). A symmetric convolutional decoder reconstructs the original signal from the latent vector. During training, 30 % of input timesteps are randomly masked.

**Patient-level feature vector (46 dimensions)**:

The encoder is applied to all windows of a patient. The per-window latent vectors are aggregated into a 32-dimensional patient descriptor: mean (8), std (8), max (8), and OLS temporal slope (8) computed across the sequence of windows. Handcrafted features are computed separately for stance (4 dims) and walking (3 dims):

- Stance: 95 % confidence ellipse sway area, lateral dominance ratio, spectral sway power (0.1–0.5 Hz), tremor power (8–12 Hz)
- Walking: normalised jerk, step interval coefficient of variation, dominant cadence frequency

Each handcrafted feature is aggregated by mean and std across windows (8 + 6 = 14 dims). Total feature vector: 32 + 14 = 46 dimensions.

**Domain adaptation**: three variants are produced for each encoder:

- `baseline`: no adaptation
- `coral`: CORAL covariance alignment — source domain covariances are whitened and re-coloured to match the target (WearPD) covariance, computed on the training set and applied to train and test
- `mmd`: MMD mean-shift — each source domain is shifted by `mu_target - mu_source`, estimated on the training set

Feature CSVs for each variant are written to `models/`.

```bash
python feature_extraction.py --arch-mode autoencoder --seed 42
python feature_extraction.py --arch-mode classifier  --seed 42
```

### Step 4a — Postural instability classification (`classifier.py`)

Binary patient-level classification (HC vs PD). Reads the feature CSVs produced in step 3 and runs a double ablation study across:

- **Feature sets**: Latent only, Handcrafted only, Combined, Combined + RFE (15 features selected by Recursive Feature Elimination with a Random Forest estimator)
- **Classifiers**: Random Forest, Gradient Boosting, SVM (RBF kernel), Logistic Regression
- **DA variants**: baseline, CORAL, MMD

The train/test split is performed at subject level, stratified by binary label (75 / 25 %). Metrics saved: accuracy, balanced accuracy, precision, recall, macro F1, ROC-AUC; per-dataset breakdowns are also reported.

```bash
python classifier.py --arch-mode autoencoder --seed 42
```

### Step 4b — Domain separability analysis (`domain_classifier.py`)

Runs the same feature sets and classifiers, but the target label is the **dataset identity** rather than the clinical label. A high domain classification accuracy indicates that the feature space retains dataset-specific information; a reduction after domain adaptation indicates successful shift mitigation. Results include accuracy, balanced accuracy, macro F1, AUC, and delta relative to baseline.

```bash
python domain_classifier.py --arch-mode autoencoder --seed 42
```

---

## Running the Full Pipeline

```bash
python run_pipeline.py --arch-mode autoencoder --seed 42
```

This executes steps 2–4 in sequence. Step 1 (`clean_data.ipynb`) must be completed manually before running the pipeline.

---

## Requirements

Python 3.12. Main dependencies:

```
tensorflow >= 2.16
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
umap-learn
openpyxl
```

Install with:

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib seaborn umap-learn openpyxl
```

---

## Configuration

All key hyperparameters are defined as module-level constants at the top of each script:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `TARGET_HZ` | `preprocessing.py` | 64 | Target sampling rate after resampling |
| `WINDOW_SEC` | `preprocessing.py` | 5 | Window duration in seconds |
| `MIN_OVERLAP` / `MAX_OVERLAP` | `preprocessing.py` | 0.50 / 0.85 | Overlap bounds for adaptive windowing |
| `TARGET_CLASS_RATIO` | `preprocessing.py` | 0.85 | Target minority-to-majority window ratio |
| `LATENT_DIM` | `feature_extraction.py` | 8 | Encoder latent dimensionality |
| `TARGET_DOMAIN` | `feature_extraction.py` | `wearpd` | Reference domain for CORAL and MMD |
| `RANDOM_STATE` | all scripts | 42 | Global random seed |

---

## Exploratory Notebooks

- **`explore_datasets.ipynb`**: distribution of windows by class, dataset, task, and subject; demographic summary
- **`results_analysis.ipynb`**: ROC curves with bootstrap confidence intervals, confusion matrices, KDE plots of latent features, UMAP visualisations, LaTeX table generation
