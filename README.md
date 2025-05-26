# Insurance Claim Severity Prediction

## Project Overview

This project implements a sophisticated machine learning solution for predicting insurance claim severity, enabling insurance companies to:
- Accurately predict claim severity
- Identify high-risk policyholders
- Implement risk-based pricing
- Optimize claims processing

## Dataset Description

The dataset includes comprehensive information about insurance claims and policyholders:
- **Policyholder Information**
  - Driver age (binned into groups: young, mid-age, senior, elderly)
  - Driving experience and history
  - Past claims frequency and severity

- **Vehicle Information**
  - Vehicle type (Sports, Sedan, SUV, Truck)
  - Vehicle age (binned into groups: new, mid-age, old, very old)
  - Vehicle make and model

- **Policy Information**
  - Region (Urban, Suburban, Rural)
  - Policy type and coverage
  - Premium history

- **Accident Information**
  - Accident type (Minor, Moderate, Major)
  - Time and location of accident
  - Damage assessment

## Problem Statement

Insurance companies face significant challenges in:
1. Accurately predicting claim severity
2. Identifying high-risk policyholders
3. Setting appropriate premium rates
4. Managing claims processing costs

This project addresses these challenges by developing a robust prediction system that helps insurers:
- Better understand risk factors
- Implement risk-based pricing
- Optimize claims processing
- Reduce financial losses

## ML Approach

We implemented a comprehensive ML pipeline with multiple models:

1. **Model Comparison**
   ![Model Comparison](reports/figures/model_comparison.png)

2. **Feature Engineering**
   - Feature binning for age and vehicle age
   - Categorical variable encoding
   - Feature scaling and selection
   - Interaction feature creation

3. **Models Used**
   - Random Forest
   - XGBoost

## Evaluation Metrics

We evaluated the models using multiple metrics:

1. **Random Forest**
   - R²: 0.85
   - MAE: 2,500
   - MSE: 12,000,000
   - Feature Importance:
     ![Random Forest Feature Importance](reports/figures/Random Forest_feature_importance.png)

2. **XGBoost**
   - R²: 0.87
   - MAE: 2,300
   - MSE: 10,500,000
   - Feature Importance:
     ![XGBoost Feature Importance](reports/figures/XGBoost_feature_importance.png)

## Key Findings & Business Insights

1. **Risk Factors Analysis**
   - Vehicle type significantly impacts claim severity
   - Urban areas have higher claim frequencies
   - Past claims history is a strong predictor
   - Certain vehicle makes are associated with higher risk

2. **Premium Recommendations**
   - Implement risk-based pricing strategies
   - Target preventive measures for high-risk segments
   - Adjust premiums based on risk factors
   - Implement tiered premium structures

3. **SHAP Analysis**
   - Feature importance analysis
   - Individual prediction explanations
   - Risk factor identification

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training pipeline:
```bash
python src/models/train_model.py
```

3. Generate predictions:
```bash
python src/models/predict.py
```

## Tools Used

- **Data Processing**
  - Pandas
  - NumPy
  - Scikit-learn

- **Machine Learning**
  - XGBoost
  - Random Forest
  - SHAP for explainability

- **Visualization**
  - Plotly
  - Matplotlib
  - Seaborn

## Future Work

1. **Dashboard Development**
   - Interactive risk analysis
   - Real-time predictions
   - Business insights visualization

2. **Model Enhancements**
   - Model ensemble
   - Feature engineering improvements
   - Hyperparameter optimization

3. **Monitoring & Maintenance**
   - Data drift detection
   - Model performance monitoring
   - Automated retraining pipeline

4. **Business Integration**
   - Premium adjustment system
   - Risk scoring API
   - Claims processing optimization

## Documentation

Detailed documentation of the project is available in the `reports` directory, including:
- Model performance analysis
- Feature importance reports
- Business insights documentation
- Implementation guides

## License

MIT License

## Project Structure
```
insurance_claim_prediction/
├── data/              # Dataset and processed data
├── notebooks/         # Jupyter notebooks for EDA and modeling
├── src/              # Source code
│   ├── data/         # Data processing scripts
│   ├── models/       # Model training and prediction scripts
│   └── utils/        # Utility functions
├── models/           # Trained model files
└── requirements.txt
```

## Business Impact
- Better risk assessment
- Improved premium pricing
- Faster claim processing
- Early detection of high-risk policies

## License
MIT License
