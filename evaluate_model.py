"""
Evaluate model performance on test data
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load model and data
print("Loading model and data...")
model = joblib.load('models/best_model.pkl')
train_df = pd.read_csv('data/train_engineered.csv')
test_df = pd.read_csv('data/test_engineered.csv')

# Prepare training data
X_train = train_df.drop(['claim_severity', 'policy_id'], axis=1, errors='ignore')
y_train = train_df['claim_severity']
y_train_log = np.log1p(y_train)

# Prepare test data
X_test = test_df.drop(['claim_severity', 'policy_id'], axis=1, errors='ignore')
y_test = test_df['claim_severity']
y_test_log = np.log1p(y_test)

# Make predictions
print("\nMaking predictions...")
y_train_pred_log = model.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)

y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)

# Calculate metrics
print("\n" + "="*60)
print("MODEL PERFORMANCE EVALUATION")
print("="*60)

print("\n📊 TRAINING SET PERFORMANCE:")
print("-" * 60)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

print(f"  MAE (Mean Absolute Error):        ${train_mae:>10,.2f}")
print(f"  RMSE (Root Mean Squared Error):   ${train_rmse:>10,.2f}")
print(f"  R² Score:                          {train_r2:>10.4f} ({train_r2*100:.2f}%)")
print(f"  MAPE (Mean Absolute % Error):      {train_mape:>10.2f}%")

print("\n📊 TEST SET PERFORMANCE:")
print("-" * 60)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print(f"  MAE (Mean Absolute Error):        ${test_mae:>10,.2f}")
print(f"  RMSE (Root Mean Squared Error):   ${test_rmse:>10,.2f}")
print(f"  R² Score:                          {test_r2:>10.4f} ({test_r2*100:.2f}%)")
print(f"  MAPE (Mean Absolute % Error):      {test_mape:>10.2f}%")

print("\n📈 PREDICTION STATISTICS:")
print("-" * 60)
print(f"  Actual Mean Claim (Train):        ${y_train.mean():>10,.2f}")
print(f"  Predicted Mean Claim (Train):     ${y_train_pred.mean():>10,.2f}")
print(f"  Actual Mean Claim (Test):         ${y_test.mean():>10,.2f}")
print(f"  Predicted Mean Claim (Test):      ${y_test_pred.mean():>10,.2f}")

print("\n✅ INTERPRETATION:")
print("-" * 60)
if test_r2 > 0.9:
    print("  🌟 EXCELLENT: Model explains >90% of variance!")
elif test_r2 > 0.7:
    print("  ✅ GOOD: Model explains >70% of variance")
elif test_r2 > 0.5:
    print("  ⚠️  MODERATE: Model explains >50% of variance")
else:
    print("  ❌ POOR: Model needs improvement")

if test_mape < 10:
    print("  🌟 EXCELLENT: Average prediction error <10%!")
elif test_mape < 20:
    print("  ✅ GOOD: Average prediction error <20%")
elif test_mape < 30:
    print("  ⚠️  MODERATE: Average prediction error <30%")
else:
    print("  ❌ HIGH ERROR: Predictions have high percentage error")

print("\n" + "="*60)
print("Model: XGBoost Regressor")
print("Features: 32 engineered features")
print("Training samples: 800")
print("Test samples: 200")
print("="*60)
