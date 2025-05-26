import os
import sys
import joblib
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import preprocess_data
from src.data.generate_synthetic_data import generate_synthetic_data

# Generate sample data
sample_data = generate_synthetic_data(n_samples=100)

# Split data into train and test sets
train_df, test_df = train_test_split(sample_data, test_size=0.2, random_state=42)

# Preprocess the data
X_train_tree, X_test_tree, y_train, y_test, X_train_linear, X_test_linear = preprocess_data(train_df, test_df)

# Load the best model
model_pipeline = joblib.load('models/best_model.pkl')

# Get the actual model from the pipeline
model = model_pipeline.named_steps['model']

# Create SHAP explainer using tree explainer since we're using XGBoost
explainer = shap.TreeExplainer(model)

# Save the explainer
joblib.dump(explainer, 'models/shap_explainer.pkl')
print("SHAP explainer created and saved successfully!")
