# ALF Health Incident Prediction - Summary Report

**Date**: August 5, 2025  
**Project**: Next-Day Health Incident Risk Prediction for Assisted Living Facility Residents

---

## Executive Summary

This project developed a predictive model to identify ALF residents at higher risk of health incidents, enabling proactive intervention by facility staff. Using 90 days of health data from 200 residents across 5 facilities, we built and evaluated machine learning models that achieve moderate predictive performance (AUC 0.608) with high specificity.

### Key Findings:
- **31.4%** of patient-days resulted in health incidents
- **Vital sign deviations** from personal baselines are the strongest predictors
- **COPD and Dementia** patients show highest incident rates (41.8% and 39.5%)
- Model identifies high-risk situations with 40.9% precision

---

## 1. Data Overview

### Dataset Characteristics:
- **18,000 records** (200 patients × 90 days)
- **26 features** including demographics, vital signs, and engineered features
- **Missing data**: 1-7% for vital sign changes (handled via median imputation)

### Target Distribution:
- No incident: 68.58%
- Incident next day: 31.42%

### High-Risk Diagnoses:
| Diagnosis | Incident Rate | Patient-Days |
|-----------|--------------|--------------|
| COPD | 41.8% | 2,700 |
| Dementia | 39.5% | 1,620 |
| Heart Disease | 37.7% | 2,430 |
| Parkinson | 36.4% | 1,620 |

---

## 2. Feature Engineering

### Key Engineered Features:
1. **Personal Baseline Deviations**: Absolute difference from patient's average vitals
2. **3-Day Rolling Averages**: Smoothed vital sign trends
3. **Daily Changes**: Day-to-day vital sign variations
4. **Risk Indicators**: Binary flags for abnormal ranges
5. **Vital Risk Score**: Composite score of risk indicators

---

## 3. Model Performance

### Model Comparison:
| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 68.5% | 66.9% |
| Precision | 50.3% | 40.9% |
| Recall | 6.4% | 11.5% |
| AUC | 0.621 | 0.608 |

The Random Forest model was selected for its better balance of precision and recall, though both models show conservative prediction patterns.

---

## 4. Top Predictive Features

### Most Important Features (Random Forest):
1. **Blood Pressure Systolic Deviation** (6.25%)
2. **Heart Rate Deviation** (6.19%)
3. **Temperature Deviation** (6.18%)
4. **Heart Rate 3-Day Average** (6.13%)
5. **BP Systolic 3-Day Average** (6.02%)

Key insight: **Deviations from personal baselines** are more predictive than absolute values.

---

## 5. Practical Recommendations

### A. Daily Monitoring Priorities
- **Monitor deviations** from personal baselines, not just absolute values
- **Red flags**:
  - Heart rate deviation >20 bpm
  - Systolic BP deviation >20 mmHg
  - Temperature deviation >1°C
  - Medication adherence <70%

### B. High-Risk Patient Profiles
1. **Primary diagnosis**: COPD, Dementia, Heart Disease
2. **Recent incident history**: Higher recurrence risk
3. **Poor medication adherence**
4. **Multiple vital sign deviations**

### C. Intervention Strategies
1. **Personalized Thresholds**: Establish individual baseline ranges
2. **Trend Monitoring**: Use 3-day rolling averages
3. **Risk Stratification**: Focus on top 20% risk scores
4. **Weekend Coverage**: Extra monitoring during weekends

### D. Implementation Guidelines
- Use model for **risk stratification**, not sole decision-making
- Combine with **clinical judgment** and patient history
- **Regular retraining** with new data (monthly recommended)
- Track model performance and adjust thresholds

---

## 6. Technical Considerations

### Model Deployment:
- Saved model and scaler for production use
- Patient-based train/test split prevents data leakage
- Handles missing values via median imputation

### Future Improvements:
1. Incorporate additional features (lab results, activity levels)
2. Experiment with time-series specific models
3. Develop patient-specific models for frequent fliers
4. Add explainability features for clinical staff

---

## Conclusion

This predictive model provides a data-driven tool for identifying at-risk ALF residents. While the model shows moderate performance, its strength lies in identifying deviations from personal baselines - a clinically meaningful insight. The model should be used as part of a comprehensive care strategy, augmenting but not replacing clinical judgment.

For optimal results, facilities should focus on establishing accurate personal baselines, ensuring consistent vital sign monitoring, and using the model's risk scores to prioritize staff attention on the highest-risk residents.