# 🛡️ Insurance Claim Severity Predictor

A comprehensive machine learning solution for predicting insurance claim severity based on policy and vehicle details. This project uses advanced feature engineering, multiple ML algorithms, and explainable AI to provide actionable insights for insurance claim assessment.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project predicts the potential severity (cost) of insurance claims using machine learning. It analyzes factors such as:
- Driver demographics (age, claim history)
- Vehicle information (type, make, age)
- Accident details (type, region)

The solution includes:
- **Data preprocessing & feature engineering** with 32+ engineered features
- **Multiple ML models** (Linear Regression, Random Forest, XGBoost)
- **Model explainability** using SHAP (SHapley Additive exPlanations)
- **Interactive web application** built with Streamlit
- **Comprehensive model evaluation** and business insights

## ✨ Features

### Machine Learning Pipeline
- ✅ Automated feature engineering with polynomial and interaction features
- ✅ Multiple model training with hyperparameter tuning
- ✅ Cross-validation and rigorous model evaluation
- ✅ Log-transformed target variable for better predictions

### Model Explainability
- 📊 SHAP waterfall plots for individual predictions
- 🔍 Feature importance analysis
- 💡 Business-friendly risk factor insights

### Web Application
- 🖥️ User-friendly Streamlit interface
- 📈 Real-time predictions with confidence metrics
- 🧠 Visual explanations of predictions
- 🎯 Actionable business insights

## 📁 Project Structure

```
insurance_claim_prediction/
│
├── app.py                          # Streamlit web application
├── evaluate_model.py               # Model evaluation script
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup configuration
├── README.md                       # Project documentation
│
├── data/                           # Data directory
│   ├── insurance_claims.csv        # Raw dataset
│   ├── train.csv                   # Training data
│   ├── test.csv                    # Test data
│   ├── train_engineered.csv        # Engineered training features
│   └── test_engineered.csv         # Engineered test features
│
├── models/                         # Saved model artifacts
│   ├── best_model.pkl              # Best performing model
│   ├── tree_preprocessor.pkl       # Preprocessing pipeline
│   ├── linear_preprocessor.pkl     # Linear model preprocessor
│   ├── shap_explainer.pkl          # SHAP explainer object
│   └── generate_shap_explainer.py  # SHAP explainer generation
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Initial data exploration
│   ├── 02_eda_analysis.ipynb       # Exploratory data analysis
│   └── 03_model_evaluation.ipynb   # Model evaluation & comparison
│
├── src/                            # Source code
│   ├── data/                       # Data processing modules
│   │   ├── preprocessing.py        # Data preprocessing
│   │   ├── feature_engineering.py  # Feature engineering pipeline
│   │   └── generate_synthetic_data.py
│   │
│   ├── models/                     # Model training & evaluation
│   │   ├── model_training.py       # Model training pipeline
│   │   ├── model_evaluation.py     # Model evaluation utilities
│   │   ├── model_explainability.py # SHAP & interpretability
│   │   └── hyperparameter_tuning.py
│   │
│   ├── business/                   # Business logic
│   │   └── business_translation.py # Convert predictions to insights
│   │
│   └── utils/                      # Utility functions
│       └── app_utils.py            # Helper functions for app
│
└── reports/                        # Generated reports
    ├── feature_importance.csv      # Feature importance scores
    ├── model_evaluation_results.csv
    ├── sample_predictions.csv
    └── figures/                    # Visualization outputs
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/YashKhandelwal0705/insurance_claim_prediction.git
cd insurance_claim_prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## 📖 Usage

### 1. Data Preparation & Feature Engineering
```bash
python src/data/generate_synthetic_data.py  # Generate synthetic data (if needed)
python src/data/feature_engineering.py      # Engineer features
```

### 2. Model Training
```bash
python src/models/model_training.py         # Train all models
python src/models/hyperparameter_tuning.py  # Fine-tune hyperparameters
```

### 3. Model Evaluation
```bash
python evaluate_model.py                    # Evaluate model performance
python src/models/model_explainability.py   # Generate SHAP explanations
```

### 4. Generate SHAP Explainer
```bash
python models/generate_shap_explainer.py    # Create SHAP explainer for app
```

### 5. Launch Web Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 Model Performance

### Best Model: XGBoost Regressor

**Test Set Performance:**
- **R² Score:** 0.9356 (93.56%)
- **MAE:** $963.07
- **RMSE:** $1,277.11
- **MAPE:** 17.80%

**Training Set Performance:**
- **R² Score:** 0.9726 (97.26%)
- **MAE:** $715.66
- **RMSE:** $919.35
- **MAPE:** 12.51%

**Prediction Statistics:**
- **Actual Mean Claim (Test):** $7,587.34
- **Predicted Mean Claim (Test):** $7,454.09

### Model Comparison

| Model              | R² Score | MAE ($) | RMSE ($) | Training Time |
|-------------------|----------|---------|----------|---------------|
| XGBoost           | 0.9356   | 963.07  | 1,277.11 | ~5 min        |
| Random Forest     | 0.9234   | 1,100+  | 1,500+   | ~3 min        |
| Linear Regression | 0.8567   | 1,500+  | 2,000+   | ~1 min        |

## 🔧 Technical Details

### Feature Engineering
The project implements comprehensive feature engineering:

1. **Binning Features:**
   - Driver age groups: Young (18-25), Mid-age (25-40), Senior (40-60), Elderly (60+)
   - Vehicle age groups: New (0-5), Mid-age (5-10), Old (10-15), Very old (15+)

2. **Polynomial Features:**
   - Second-degree polynomials for numerical features
   - Interaction terms between key variables

3. **Interaction Features:**
   - Age × Vehicle type
   - Vehicle age × Make
   - Driver age × Vehicle make

4. **Total Features:** 32 engineered features

### Preprocessing Pipeline
- **Categorical encoding:** One-Hot Encoding
- **Numerical scaling:** StandardScaler (for linear models)
- **Target transformation:** Log transformation (log1p)

### Model Architecture
```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])),
    ('model', XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    ))
])
```

## 🖥️ Web Application

The Streamlit application provides:

### Features:
1. **Interactive Input Form:**
   - Driver age slider (18-80)
   - Past claims counter
   - Vehicle details (type, make, age)
   - Accident type and region selectors

2. **Prediction Dashboard:**
   - Estimated claim amount with currency formatting
   - Key risk factors with icons
   - Business-friendly insights

3. **Explainability Tab:**
   - SHAP waterfall plot
   - Feature contribution analysis
   - Visual explanation of prediction drivers

### Running the App:
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the interface.

## 🧪 Testing

Run model evaluation:
```bash
python evaluate_model.py
```

Expected output:
```
MODEL PERFORMANCE EVALUATION
============================================================

📊 TEST SET PERFORMANCE:
------------------------------------------------------------
  MAE (Mean Absolute Error):        $    963.07
  RMSE (Root Mean Squared Error):   $  1,277.11
  R² Score:                              0.9356 (93.56%)
  MAPE (Mean Absolute % Error):           17.80%

✅ INTERPRETATION:
------------------------------------------------------------
  🌟 EXCELLENT: Model explains >90% of variance!
  ✅ GOOD: Average prediction error <20%
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Yash Khandelwal**
- GitHub: [@YashKhandelwal0705](https://github.com/YashKhandelwal0705)

## 🙏 Acknowledgments

- XGBoost library for high-performance gradient boosting
- SHAP library for model interpretability
- Streamlit for the web application framework
- scikit-learn for ML utilities and preprocessing

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact the author.

---

⭐ If you find this project helpful, please consider giving it a star!
