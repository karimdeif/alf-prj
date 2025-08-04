# ALF Health Incident Prediction Project

## Overview
This project implements a machine learning solution to predict next-day health incidents for Assisted Living Facility (ALF) residents. The goal is to help facility managers proactively identify at-risk patients and allocate resources effectively.

## Project Structure
```
alf_project/
├── generate_alf_data.py          # Synthetic data generation script
├── alf_health_data.csv           # Generated dataset (18,000 records)
├── alf_health_analysis.py        # Main analysis script
├── ALF_Health_Analysis_Notebook.py # Complete analysis notebook
├── ALF_Health_Incident_Prediction_Report.md  # Summary report
├── rf_model.pkl                  # Trained Random Forest model
├── scaler.pkl                    # Feature scaler
├── exploration_plots.png         # Data exploration visualizations
├── roc_curves.png               # Model ROC curves
├── confusion_matrices.png        # Model confusion matrices
├── feature_importance.png        # Feature importance plot
└── README.md                     # This file
```

## Key Results
- **Dataset**: 200 patients, 90 days, 18,000 records
- **Target**: 31.4% incident rate
- **Best Model**: Random Forest (AUC: 0.608)
- **Top Features**: Vital sign deviations from personal baselines

## Quick Start

### 1. View the Analysis
```bash
# Activate virtual environment
source bin/activate

# Run the complete analysis
python alf_health_analysis.py

# Or run the notebook version
python ALF_Health_Analysis_Notebook.py
```

### 2. Read the Report
Open `ALF_Health_Incident_Prediction_Report.md` for a comprehensive summary of findings and recommendations.

### 3. Use the Model
```python
import pickle
import pandas as pd

# Load model and scaler
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare features (same 26 features used in training)
# Scale and predict
X_scaled = scaler.transform(X)
predictions = model.predict_proba(X_scaled)[:, 1]
```

## Key Recommendations
1. **Monitor deviations** from personal baselines, not just absolute values
2. **Focus on high-risk groups**: COPD (41.8%) and Dementia (39.5%) patients
3. **Red flags**: HR deviation >20 bpm, BP deviation >20 mmHg, Temp >1°C
4. Use model for **risk stratification**, combined with clinical judgment

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

All dependencies are installed in the virtual environment.

## Author
Data Science Team - August 2025