import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_exploratory_analysis():

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(repo_root, 'data', 'extracted_features', 'features_5s_256.csv')
    folder = os.path.join(repo_root, 'data', 'cleaned_data')
    
    print(f"Loading features from: {features_path}")
    features = pd.read_csv(features_path, dtype={'subjectID': str}, low_memory=False)
    
    # Load and combine clinical data files
    clinical_data = pd.DataFrame()
    for file in os.listdir(folder):
        if file.endswith('clinical.csv') or file.endswith('clinical_data.csv'):
            df = pd.read_csv(os.path.join(folder, file), dtype={'subjectID': str})
            # Extract dataset name from filename
            df['dataset'] = re.sub(r'_clinical(?:_data)?\.csv$', '', file)
            clinical_data = pd.concat([clinical_data, df], ignore_index=True)

    # Clean IDs for matching
    features['subjectID'] = features['subjectID'].astype(str).str.strip()
    clinical_data['subjectID'] = clinical_data['subjectID'].astype(str).str.strip()
    
    # Merge datasets
    merged_data = pd.merge(features, clinical_data, on=['subjectID', 'dataset'])
    
    # --- 3. Preprocessing ---
    # Exclude specific datasets if needed (e.g., fog_star) and drop NaNs
    analysis_df = merged_data[merged_data['dataset'] != 'fog_star'].copy()
    analysis_df = analysis_df.dropna(subset=['postural_stability'])
    
    # Binary Target Creation: 0 (Stable/Mild) vs 1 (Moderate/Severe)
    analysis_df["binary_ps"] = analysis_df['postural_stability'].apply(lambda x: 0 if x <= 1 else 1)
    
    # Identify feature columns (assuming they are numeric strings from 0 to N)
    # We filter columns that are purely numeric names
    feature_cols = [col for col in analysis_df.columns if col.isdigit()]
    
    X = analysis_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y = analysis_df['binary_ps']
    
    # Scaling is mandatory for PCA and t-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 4. Dimensionality Reduction ---
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print("Running t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # --- 5. Plotting ---
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # PCA Plot
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.6, ax=ax1)
    ax1.set_title('PCA Projection: Linear Feature Variance', fontsize=14)
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend(title='Binary PS (0=Stable, 1=Unstable)')

    # t-SNE Plot
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='coolwarm', alpha=0.6, ax=ax2)
    ax2.set_title(f't-SNE Visualization: Non-Linear Clustering\n(Features: {len(feature_cols)}, Window: 5s)', fontsize=14)
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(title='Binary PS (0=Stable, 1=Unstable)')

    plt.suptitle('Exploratory Data Analysis of HUF Features for Postural Instability', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot for your thesis
    plt.savefig('huf_feature_distribution.png', dpi=300)
    print("Plot saved as huf_feature_distribution.png")
    plt.show()

if __name__ == "__main__":
    run_exploratory_analysis()