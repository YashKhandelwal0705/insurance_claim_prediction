import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load test data
test_df = pd.read_csv('data/test_engineered.csv')
X_test = test_df.drop('claim_severity', axis=1)
y_test = np.log1p(test_df['claim_severity'])  # Using log-transformed target

# Load all trained models
models = {}
models['Linear Regression'] = joblib.load('models/linear_regression.pkl')
models['Random Forest'] = joblib.load('models/random_forest.pkl')
models['XGBoost'] = joblib.load('models/xgboost.pkl')

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate model performance on test set"""
    print(f'\nEvaluating {model_name}...')
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R² Score': r2
    }

# Evaluate all models and store results
results = {}
for model_name, model in models.items():
    results[model_name] = evaluate_model(model_name, model, X_test, y_test)

# Create a DataFrame for better visualization
results_df = pd.DataFrame(results).T
results_df = results_df[['MAE', 'RMSE', 'R² Score']]

# Plot model performance comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MAE comparison
sns.barplot(x=results_df.index, y='MAE', data=results_df, ax=axes[0])
axes[0].set_title('MAE Comparison')
axes[0].set_ylabel('MAE')
axes[0].set_xlabel('Model')

# RMSE comparison
sns.barplot(x=results_df.index, y='RMSE', data=results_df, ax=axes[1])
axes[1].set_title('RMSE Comparison')
axes[1].set_ylabel('RMSE')
axes[1].set_xlabel('Model')

# R² Score comparison
sns.barplot(x=results_df.index, y='R² Score', data=results_df, ax=axes[2])
axes[2].set_title('R² Score Comparison')
axes[2].set_ylabel('R² Score')
axes[2].set_xlabel('Model')

plt.tight_layout()
plt.savefig('reports/figures/model_comparison.png')
plt.close()

# Save the best model
best_model = models['XGBoost']
joblib.dump(best_model, 'models/best_model.pkl')

# Print final results
print("\nFinal Model Performance Summary:")
print("-" * 50)
print(results_df)

# Save results to CSV
results_df.to_csv('reports/model_evaluation_results.csv')
