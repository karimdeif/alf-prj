"""
Generate synthetic ALF (Assisted Living Facility) health data
for predicting next-day health incidents
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_alf_data(n_patients=200, n_days=90, n_facilities=5):
    """
    Generate synthetic health data for ALF residents
    
    Parameters:
    -----------
    n_patients : int
        Number of unique patients
    n_days : int
        Number of days of data per patient
    n_facilities : int
        Number of facilities
    
    Returns:
    --------
    pd.DataFrame : Synthetic health data
    """
    
    # Define possible diagnoses and their base incident rates
    diagnoses = {
        'Dementia': 0.15,
        'Diabetes': 0.12,
        'Heart Disease': 0.18,
        'COPD': 0.20,
        'Hypertension': 0.10,
        'Arthritis': 0.08,
        'Depression': 0.11,
        'Parkinson': 0.16
    }
    
    # Generate patient demographics
    patients = []
    for i in range(n_patients):
        patient = {
            'patient_id': f'P{i+1:04d}',
            'facility_id': f'F{random.randint(1, n_facilities):03d}',
            'age': np.random.normal(78, 8),  # Average age ~78
            'gender': random.choice(['Male', 'Female']),
            'diagnosis': random.choice(list(diagnoses.keys())),
            'baseline_hr': np.random.normal(75, 10),
            'baseline_bp_sys': np.random.normal(130, 15),
            'baseline_bp_dia': np.random.normal(80, 10),
            'baseline_temp': np.random.normal(36.8, 0.3)
        }
        patient['age'] = max(65, min(95, int(patient['age'])))
        patients.append(patient)
    
    # Generate daily health records
    records = []
    start_date = datetime(2024, 1, 1)
    
    for patient in patients:
        base_incident_rate = diagnoses[patient['diagnosis']]
        
        # Age factor (older patients have higher risk)
        age_factor = 1 + (patient['age'] - 75) * 0.02
        
        # Gender factor (slight difference)
        gender_factor = 1.1 if patient['gender'] == 'Male' else 1.0
        
        # Track previous vitals for trend calculation
        prev_hr = patient['baseline_hr']
        prev_bp_sys = patient['baseline_bp_sys']
        prev_bp_dia = patient['baseline_bp_dia']
        prev_temp = patient['baseline_temp']
        prev_incident = False
        
        for day in range(n_days):
            current_date = start_date + timedelta(days=day)
            
            # Generate vitals with some correlation to previous day
            # and increased variation if previous incident
            variation_factor = 1.5 if prev_incident else 1.0            
            heart_rate = prev_hr + np.random.normal(0, 5 * variation_factor)
            heart_rate = max(50, min(120, heart_rate))
            
            bp_sys = prev_bp_sys + np.random.normal(0, 8 * variation_factor)
            bp_sys = max(90, min(180, bp_sys))
            
            bp_dia = prev_bp_dia + np.random.normal(0, 5 * variation_factor)
            bp_dia = max(60, min(110, bp_dia))
            
            temperature = prev_temp + np.random.normal(0, 0.3 * variation_factor)
            temperature = max(35.5, min(39.0, temperature))
            
            # Medication adherence (tends to be lower after incidents)
            if prev_incident:
                med_adherence = np.random.beta(7, 3)  # Skewed lower
            else:
                med_adherence = np.random.beta(9, 2)  # Skewed higher
            
            # Additional features that affect incident risk
            hr_deviation = abs(heart_rate - patient['baseline_hr'])
            bp_sys_deviation = abs(bp_sys - patient['baseline_bp_sys'])
            temp_deviation = abs(temperature - patient['baseline_temp'])
            
            # Calculate incident probability based on multiple factors
            incident_prob = base_incident_rate * age_factor * gender_factor
            
            # Increase risk based on vital deviations
            if hr_deviation > 20:
                incident_prob *= 1.5
            if bp_sys_deviation > 20:
                incident_prob *= 1.4
            if temp_deviation > 1:
                incident_prob *= 1.8
            if med_adherence < 0.7:
                incident_prob *= 1.3
            
            # Previous incident increases risk
            if prev_incident:
                incident_prob *= 1.6
            
            # Cap probability
            incident_prob = min(0.5, incident_prob)
            
            # Determine if incident occurs next day
            incident_next_day = 1 if random.random() < incident_prob else 0
            
            # Add some missing values randomly (realistic data)
            if random.random() < 0.02:  # 2% missing values
                heart_rate = np.nan
            if random.random() < 0.03:  # 3% missing BP
                bp_sys = np.nan
                bp_dia = np.nan
            if random.random() < 0.01:  # 1% missing temp
                temperature = np.nan
            
            record = {
                'patient_id': patient['patient_id'],
                'facility_id': patient['facility_id'],
                'date': current_date.strftime('%Y-%m-%d'),
                'age': patient['age'],
                'gender': patient['gender'],
                'diagnosis': patient['diagnosis'],
                'heart_rate': round(heart_rate, 1) if not pd.isna(heart_rate) else np.nan,
                'blood_pressure_sys': round(bp_sys, 1) if not pd.isna(bp_sys) else np.nan,
                'blood_pressure_dia': round(bp_dia, 1) if not pd.isna(bp_dia) else np.nan,
                'temperature': round(temperature, 1) if not pd.isna(temperature) else np.nan,
                'med_adherence': round(med_adherence, 2),
                'incident_next_day': incident_next_day
            }
            
            records.append(record)
            
            # Update previous values
            prev_hr = heart_rate if not pd.isna(heart_rate) else prev_hr
            prev_bp_sys = bp_sys if not pd.isna(bp_sys) else prev_bp_sys
            prev_bp_dia = bp_dia if not pd.isna(bp_dia) else prev_bp_dia
            prev_temp = temperature if not pd.isna(temperature) else prev_temp
            prev_incident = bool(incident_next_day)
    
    return pd.DataFrame(records)

def add_derived_features(df):
    """
    Add derived features to the dataset
    """
    # Sort by patient and date
    df = df.sort_values(['patient_id', 'date']).reset_index(drop=True)
    
    # Calculate rolling averages and changes for each patient
    patient_groups = df.groupby('patient_id')
    
    # 3-day rolling averages
    df['heart_rate_3d_avg'] = patient_groups['heart_rate'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['bp_sys_3d_avg'] = patient_groups['blood_pressure_sys'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['temp_3d_avg'] = patient_groups['temperature'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Daily changes
    df['heart_rate_change'] = patient_groups['heart_rate'].diff()
    df['bp_sys_change'] = patient_groups['blood_pressure_sys'].diff()
    df['temp_change'] = patient_groups['temperature'].diff()
    
    # Vital signs risk indicators
    df['high_hr'] = (df['heart_rate'] > 100).astype(int)
    df['low_hr'] = (df['heart_rate'] < 60).astype(int)
    df['high_bp'] = (df['blood_pressure_sys'] > 140).astype(int)
    df['low_bp'] = (df['blood_pressure_sys'] < 90).astype(int)
    df['fever'] = (df['temperature'] > 37.5).astype(int)
    df['low_med_adherence'] = (df['med_adherence'] < 0.8).astype(int)
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 70, 80, 90, 100], 
                            labels=['65-70', '71-80', '81-90', '90+'])
    
    # Days since last incident (for each patient)
    df['days_since_incident'] = 0
    for patient_id in df['patient_id'].unique():
        patient_mask = df['patient_id'] == patient_id
        patient_data = df[patient_mask].copy()
        
        days_counter = 0
        days_since = []
        
        for idx, row in patient_data.iterrows():
            days_since.append(days_counter)
            if row['incident_next_day'] == 1:
                days_counter = 0
            else:
                days_counter += 1
        
        df.loc[patient_mask, 'days_since_incident'] = days_since
    
    return df


if __name__ == "__main__":
    # Generate the data
    print("Generating synthetic ALF health data...")
    df = generate_alf_data(n_patients=200, n_days=90, n_facilities=5)
    
    # Add derived features
    print("Adding derived features...")
    df = add_derived_features(df)
    
    # Save to CSV
    output_path = '/home/karimdeif/alf_project/alf_health_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total records: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Incident rate: {df['incident_next_day'].mean():.2%}")
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])