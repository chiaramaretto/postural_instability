import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import re


def load_features_csv(path):
    try:
        return pd.read_csv(path, dtype={'subjectID': str}, low_memory=False)
    except (ValueError, TypeError):
        # Fallback for mixed-type columns that can break strict parser conversion.
        return pd.read_csv(path, dtype=str, low_memory=False)

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(repo_root, 'data', 'extracted_features', 'features_5s_256.csv')
folder = os.path.join(repo_root, 'data', 'cleaned_data')

print(f"Loading features from: {features_path}")
features = load_features_csv(features_path)

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

# 1. Prepare Data (Assumes you have analysis_df from the previous script)
# feature_cols are the 256/64 columns from HUF
X = analysis_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
y = analysis_df['binary_ps'].astype(int)

# 2. Train Random Forest for Importance Extraction
print("Training Random Forest to evaluate feature importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X, y)

# 3. Get Feature Importances
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 4. Plot Top 20 Features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis')
plt.title('Top 20 HUF Features by Random Forest Importance', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('HUF Feature Index')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# 5. Analysis of results
top_importance = feature_importance_df.iloc[0]['Importance']
bottom_importance = feature_importance_df.iloc[-1]['Importance']
ratio = top_importance / bottom_importance

print(f"Total features analyzed: {len(feature_names)}")
print(f"Most Important Feature: {feature_importance_df.iloc[0]['Feature']} (Score: {top_importance:.4f})")
print(f"Importance Ratio (Top/Bottom): {ratio:.2f}")