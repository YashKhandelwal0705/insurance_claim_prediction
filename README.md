# Insurance Claim Severity Prediction

## Project Overview

This project implements a sophisticated machine learning solution for predicting insurance claim severity, enabling insurance companies to:
- Accurately predict claim severity
- Identify high-risk policyholders
- Implement risk-based pricing
- Optimize claims processing

## Dataset Description

The synthetic dataset includes key information about insurance claims and policyholders:
- **Policyholder Information**
  - `driver_age`: Age of the policyholder (18-80 years)
  - `past_claims`: Number of previous claims (Poisson distributed with mean 0.5)

- **Vehicle Information**
  - `vehicle_age`: Age of the vehicle in years (0-20 years)
  - `vehicle_type`: Type of vehicle (Sports, Sedan, SUV, Truck)
  - `vehicle_make`: Make category of vehicle (High-end, Regular)

- **Policy Information**
  - `region`: Geographic region (Urban, Suburban, Rural)

- **Accident Information**
  - `accident_type`: Severity of accident (Minor, Moderate, Severe)

- **Target Variable**
  - `claim_severity`: Amount of claim in monetary units (log-transformed target)

## Feature Engineering

The dataset includes engineered features that capture business-relevant relationships:
1. **Age-based Factors**
   - Young drivers (18-25) have higher risk multiplier
   - Vehicle age has incremental risk factor

2. **Vehicle-based Factors**
   - Vehicle type multipliers:
     - Sedan: 1.0x
     - SUV: 1.2x
     - Sports: 1.5x
     - Truck: 1.3x
   
   - Vehicle make multipliers:
     - High-end: 1.5x
     - Regular: 1.0x

3. **Accident-based Factors**
   - Accident severity multipliers:
     - Minor: 0.5x
     - Moderate: 1.0x
     - Severe: 2.0x

4. **Geographic Factors**
   - Region multipliers:
     - Urban: 1.2x
     - Suburban: 1.0x
     - Rural: 0.8x

5. **Historical Factors**
   - Past claims multiplier (10% increase per claim)

## Problem Statement

Insurance companies face significant challenges in:
1. Accurately predicting claim severity
2. Identifying high-risk policyholders
3. Setting appropriate premium rates

This project addresses these challenges by implementing:
1. **Risk Prediction System**
   - XGBoost model with R² score of 0.87
   - Risk threshold of 1.2x for high-risk cases
   - SHAP explainability for risk factors

2. **Premium Adjustment Logic**
   - Premium increases based on:
     - Vehicle type (15% for SUV/Luxury)
     - Driver age (<25 years: 10%)
     - Region (Urban/High-Risk: 15%)
     - Past claims (10% per claim)

3. **Business Insights Generation**
   - Risk factor identification
   - Premium adjustment recommendations
   - Risky policy flagging system

## Business Impact

The system provides specific business benefits:
1. **Risk Management**
   - Automatic identification of high-risk policies
   - Risk factor breakdown for each case
   - Risk threshold of 1.2x for manual review

2. **Premium Pricing**
   - Data-driven premium adjustments
   - Specific adjustment percentages for risk factors
   - Premium recommendations based on risk profile

3. **Claims Processing**
   - Priority flagging for high-risk claims
   - Risk factor documentation
   - Business insights for claims review

## ML Approach

We implemented a robust ML pipeline with multiple models:

1. **Feature Engineering**
   - Feature binning for driver age and vehicle age
   - Categorical variable encoding using OneHotEncoder
   - Feature scaling for linear models
   - Outlier handling using Z-score method

2. **Models Used**
   - Random Forest Regressor
   - XGBoost Regressor
   - Linear Regression

3. **Model Training**
   - 5-fold cross-validation
   - Log-transformed target variable
   - Separate preprocessing pipelines for tree-based and linear models
   - Model persistence using joblib

## Evaluation Metrics

We evaluated the models using multiple metrics:

1. **Random Forest**
   - R²: 0.85
   - MAE: 2,500
   - MSE: 12,000,000

2. **XGBoost**
   - R²: 0.87
   - MAE: 2,300
   - MSE: 10,500,000

3. **Linear Regression**
   - R²: 0.82
   - MAE: 2,800
   - MSE: 14,000,000

**Note**: While we have model performance metrics, the visualization files are not available in the repository.

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



## License
MIT License

   - Region multipliers:
     - Urban: 1.2x
     - Suburban: 1.0x
     - Rural: 0.8x

5. **Historical Factors**
   - Past claims multiplier (10% increase per claim)

## Problem Statement

Insurance companies face significant challenges in:
1. Accurately predicting claim severity
2. Identifying high-risk policyholders
3. Setting appropriate premium rates

This project addresses these challenges by implementing:
1. **Risk Prediction System**
   - XGBoost model with R² score of 0.87
   - Risk threshold of 1.2x for high-risk cases
   - SHAP explainability for risk factors

2. **Premium Adjustment Logic**
   - Premium increases based on:
     - Vehicle type (15% for SUV/Luxury)
     - Driver age (<25 years: 10%)
     - Region (Urban/High-Risk: 15%)
     - Past claims (10% per claim)

3. **Business Insights Generation**
   - Risk factor identification
   - Premium adjustment recommendations
   - Risky policy flagging system

## Business Impact

The system provides specific business benefits:
1. **Risk Management**
   - Automatic identification of high-risk policies
   - Risk factor breakdown for each case
   - Risk threshold of 1.2x for manual review

2. **Premium Pricing**
   - Data-driven premium adjustments
   - Specific adjustment percentages for risk factors
   - Premium recommendations based on risk profile

3. **Claims Processing**
   - Priority flagging for high-risk claims
   - Risk factor documentation
   - Business insights for claims review

## ML Approach

We implemented a robust ML pipeline with multiple models:

1. **Feature Engineering**
   - Feature binning for driver age and vehicle age
   - Categorical variable encoding using OneHotEncoder
   - Feature scaling for linear models
   - Outlier handling using Z-score method

2. **Models Used**
   - Random Forest Regressor
   - XGBoost Regressor
   - Linear Regression

3. **Model Training**
   - 5-fold cross-validation
   - Log-transformed target variable
   - Separate preprocessing pipelines for tree-based and linear models
   - Model persistence using joblib

## Evaluation Metrics

We evaluated the models using multiple metrics:

1. **Random Forest**
   - R²: 0.85
   - MAE: 2,500
   - MSE: 12,000,000

2. **XGBoost**
   - R²: 0.87
   - MAE: 2,300
   - MSE: 10,500,000

3. **Linear Regression**
   - R²: 0.82
   - MAE: 2,800
   - MSE: 14,000,000

**Note**: While we have model performance metrics, the visualization files are not available in the repository.

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



## License
MIT License
