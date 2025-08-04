#!/usr/bin/env python
# coding: utf-8

"""
ALF Health Incident Prediction - Complete Analysis Notebook
==========================================================

This notebook contains the complete pipeline for predicting next-day health incidents
in Assisted Living Facility (ALF) residents.

Author: Data Science Team
Date: August 5, 2025
"""

# %% [markdown]
# # ALF Health Incident Prediction
# 
# ## Project Overview
# This project aims to help facility managers identify patients at higher risk of health incidents,
# enabling proactive intervention and better care management.

# %% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style and random seed
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

# %% [markdown]
# ## 1. Data Loading and Initial Exploration
# %% Load Data
print("Loading ALF health data...")
df = pd.read_csv('alf_health_data.csv')
print(f"✓ Data loaded: {df.shape[0]:,} records, {df.shape[1]} features")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# %% Data Overview
print("\n📊 Dataset Overview:")
print("-" * 50)
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Unique facilities: {df['facility_id'].nunique()}")
print(f"\nTarget distribution:")
print(f"  • No incident: {(df['incident_next_day'] == 0).sum():,} ({(df['incident_next_day'] == 0).mean():.1%})")
print(f"  • Incident: {(df['incident_next_day'] == 1).sum():,} ({(df['incident_next_day'] == 1).mean():.1%})")

# %% Missing Values Analysis
missing_summary = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_summary) > 0:
    print("\n⚠️  Missing Values:")
    print(missing_summary)
# %% [markdown]
# ## 2. Exploratory Data Analysis

# %% Incident Rate by Diagnosis
diagnosis_stats = df.groupby('diagnosis')['incident_next_day'].agg(['mean', 'count']).round(3)
diagnosis_stats.columns = ['Incident_Rate', 'Patient_Days']
diagnosis_stats = diagnosis_stats.sort_values('Incident_Rate', ascending=False)

print("\n🏥 Incident Rate by Diagnosis:")
print(diagnosis_stats)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart of incident rates
diagnosis_stats['Incident_Rate'].plot(kind='bar', ax=ax1, color='coral')
ax1.set_title('Incident Rate by Diagnosis', fontsize=14, fontweight='bold')
ax1.set_xlabel('Diagnosis')
ax1.set_ylabel('Incident Rate')
ax1.axhline(y=df['incident_next_day'].mean(), color='black', linestyle='--', alpha=0.5, label='Overall Average')
ax1.legend()

# Distribution of vital signs by incident status
vital_cols = ['heart_rate', 'blood_pressure_sys', 'temperature']
for i, col in enumerate(vital_cols):
    df[df['incident_next_day'] == 0][col].hist(bins=30, alpha=0.5, label='No Incident', ax=ax2)
    
df[df['incident_next_day'] == 1]['heart_rate'].hist(bins=30, alpha=0.5, label='Incident', ax=ax2, color='red')
ax2.set_title('Heart Rate Distribution by Incident Status', fontsize=14, fontweight='bold')
ax2.set_xlabel('Heart Rate (bpm)')
ax2.legend()

plt.tight_layout()
plt.show()
# %% [markdown]
# ## 3. Feature Engineering

# %% Create Modeling Dataset
df_model = df.copy()

# Time-based features
df_model['day_of_week'] = df_model['date'].dt.dayofweek
df_model['is_weekend'] = (df_model['day_of_week'].isin([5, 6])).astype(int)

# Calculate personal baselines
patient_baselines = df_model.groupby('patient_id')[['heart_rate', 'blood_pressure_sys', 
                                                    'blood_pressure_dia', 'temperature']].mean()
patient_baselines.columns = [col + '_baseline' for col in patient_baselines.columns]
df_model = df_model.merge(patient_baselines, left_on='patient_id', right_index=True)

# Deviation features
df_model['hr_deviation'] = abs(df_model['heart_rate'] - df_model['heart_rate_baseline'])
df_model['bp_sys_deviation'] = abs(df_model['blood_pressure_sys'] - df_model['blood_pressure_sys_baseline'])
df_model['temp_deviation'] = abs(df_model['temperature'] - df_model['temperature_baseline'])

# Composite risk score
df_model['vital_risk_score'] = (
    df_model['high_hr'] + df_model['low_hr'] + 
    df_model['high_bp'] + df_model['low_bp'] + 
    df_model['fever'] + df_model['low_med_adherence']
)

print("✓ Feature engineering complete!")
print(f"  • Added {len([col for col in df_model.columns if col not in df.columns])} new features")
# %% [markdown]
# ## 4. Model Training and Evaluation

# %% Prepare Features and Handle Missing Values
# Encode categorical variables
le_gender = LabelEncoder()
le_diagnosis = LabelEncoder()
df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'])
df_model['diagnosis_encoded'] = le_diagnosis.fit_transform(df_model['diagnosis'])
df_model['age_group_encoded'] = df_model['age_group'].map({'65-70': 0, '71-80': 1, '81-90': 2, '90+': 3})

# Select features
feature_cols = [
    'age', 'gender_encoded', 'diagnosis_encoded', 'heart_rate', 'blood_pressure_sys',
    'blood_pressure_dia', 'temperature', 'heart_rate_3d_avg', 'bp_sys_3d_avg',
    'temp_3d_avg', 'heart_rate_change', 'bp_sys_change', 'temp_change',
    'hr_deviation', 'bp_sys_deviation', 'temp_deviation', 'high_hr', 'low_hr',
    'high_bp', 'low_bp', 'fever', 'low_med_adherence', 'vital_risk_score',
    'med_adherence', 'days_since_incident', 'is_weekend'
]

# Handle missing values
numeric_cols = [col for col in feature_cols if col in df_model.select_dtypes(include=[np.number]).columns]
imputer = SimpleImputer(strategy='median')
df_model[numeric_cols] = imputer.fit_transform(df_model[numeric_cols])

X = df_model[feature_cols]
y = df_model['incident_next_day']
# %% Train-Test Split (By Patient)
# CRITICAL: Split by patient to avoid data leakage
patient_ids = df_model['patient_id']
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, patient_ids))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("🔄 Train-Test Split (by patient):")
print(f"  • Training: {X_train.shape[0]:,} records from {patient_ids.iloc[train_idx].nunique()} patients")
print(f"  • Testing: {X_test.shape[0]:,} records from {patient_ids.iloc[test_idx].nunique()} patients")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Model Training
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}
print("\n🤖 Training Models...")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"\n{name}:")
    print(f"  • Accuracy: {results[name]['accuracy']:.3f}")
    print(f"  • Precision: {results[name]['precision']:.3f}")
    print(f"  • Recall: {results[name]['recall']:.3f}")
    print(f"  • AUC: {results[name]['auc']:.3f}")
# %% [markdown]
# ## 5. Model Evaluation and Feature Importance

# %% ROC Curves
plt.figure(figsize=(10, 8))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {result['auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Health Incident Prediction', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

# %% Feature Importance
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n🎯 Top 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.3f}")
# %% [markdown]
# ## 6. Key Insights and Recommendations
# 
# ### 🔍 Key Findings:
# 1. **Vital sign deviations from personal baselines** are the strongest predictors
# 2. **COPD and Dementia patients** have the highest incident rates
# 3. Model achieves **60.8% AUC** with conservative but specific predictions
# 
# ### 💡 Practical Recommendations:
# 
# **Daily Monitoring Priorities:**
# - Monitor deviations from personal baselines (not just absolute values)
# - Red flags: HR deviation >20 bpm, BP deviation >20 mmHg, Temp deviation >1°C
# 
# **High-Risk Patient Profiles:**
# - COPD patients (41.8% incident rate)
# - Dementia patients (39.5% incident rate)
# - Recent incident history
# - Low medication adherence (<70%)
# 
# **Implementation Strategy:**
# - Use model for risk stratification, not sole decision-making
# - Focus on top 20% risk scores for intervention
# - Establish personalized baseline thresholds
# - Regular model retraining with new data

# %% Save Model and Scaler
import pickle

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Analysis complete! Model and scaler saved for deployment.")