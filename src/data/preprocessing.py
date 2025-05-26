import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data():
    """Load the synthetic dataset"""
    train_df = pd.read_csv(r'F:\Projects\insurance_claim_prediction\data\train.csv')
    test_df = pd.read_csv(r'F:\Projects\insurance_claim_prediction\data\test.csv')
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Preprocess the data:
    1. Handle categorical variables
    2. Scale numerical features
    3. Handle outliers
    """
    # Separate features and target
    X_train = train_df.drop(['claim_severity', 'policy_id'], axis=1)
    y_train = train_df['claim_severity']
    
    X_test = test_df.drop(['claim_severity', 'policy_id'], axis=1)
    y_test = test_df['claim_severity']
    
    # Define categorical and numerical columns
    categorical_cols = ['vehicle_type', 'vehicle_make', 'accident_type', 'region']
    numerical_cols = ['driver_age', 'vehicle_age', 'past_claims']
    
    # Create preprocessing pipelines
    # For tree-based models (RandomForest, XGBoost)
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )
    
    # For linear models (Linear Regression)
    linear_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )
    
    # Apply preprocessing
    X_train_tree = tree_preprocessor.fit_transform(X_train)
    X_test_tree = tree_preprocessor.transform(X_test)
    
    X_train_linear = linear_preprocessor.fit_transform(X_train)
    X_test_linear = linear_preprocessor.transform(X_test)
    
    # Save preprocessing objects
    os.makedirs(r'F:\Projects\insurance_claim_prediction\models', exist_ok=True)
    joblib.dump(tree_preprocessor, r'F:\Projects\insurance_claim_prediction\models\tree_preprocessor.pkl')
    joblib.dump(linear_preprocessor, r'F:\Projects\insurance_claim_prediction\models\linear_preprocessor.pkl')
    
    return (
        X_train_tree, X_test_tree, y_train, y_test,
        X_train_linear, X_test_linear
    )

def handle_outliers(X_train, y_train):
    """
    Handle outliers using Z-score method
    Only applied to numerical features
    """
    numerical_cols = ['driver_age', 'vehicle_age', 'past_claims']
    
    # Calculate Z-scores
    z_scores = np.abs((X_train[numerical_cols] - X_train[numerical_cols].mean()) / X_train[numerical_cols].std())
    
    # Identify outliers (Z-score > 3)
    outliers = (z_scores > 3).any(axis=1)
    
    # Remove outliers from training data
    X_train_no_outliers = X_train[~outliers]
    y_train_no_outliers = y_train[~outliers]
    
    return X_train_no_outliers, y_train_no_outliers

def create_feature_bins(X_train, X_test):
    """
    Create bins for numerical features
    """
    # Create age bins
    bins = [18, 25, 40, 60, 80]
    labels = ['young', 'mid_age', 'senior', 'elderly']
    
    X_train['age_group'] = pd.cut(X_train['driver_age'], bins=bins, labels=labels)
    X_test['age_group'] = pd.cut(X_test['driver_age'], bins=bins, labels=labels)
    
    # Create vehicle age bins
    bins = [0, 5, 10, 15, 20]
    labels = ['new', 'mid_age', 'old', 'very_old']
    
    X_train['vehicle_age_group'] = pd.cut(X_train['vehicle_age'], bins=bins, labels=labels)
    X_test['vehicle_age_group'] = pd.cut(X_test['vehicle_age'], bins=bins, labels=labels)
    
    return X_train, X_test

def main():
    """Main function to execute preprocessing pipeline"""
    # Load data
    train_df, test_df = load_data()
    
    # Create feature bins
    train_df, test_df = create_feature_bins(train_df, test_df)
    
    # Handle outliers
    X_train = train_df.drop(['claim_severity'], axis=1)
    y_train = train_df['claim_severity']
    X_train_no_outliers, y_train_no_outliers = handle_outliers(X_train, y_train)
    
    # Preprocess data
    X_train_tree, X_test_tree, y_train, y_test, \
    X_train_linear, X_test_linear = preprocess_data(
        pd.concat([X_train_no_outliers, y_train_no_outliers], axis=1),
        test_df
    )
    
    print("Data preprocessing completed!")
    print(f"Training samples after preprocessing: {len(X_train_tree)}")
    print(f"Test samples: {len(X_test_tree)}")

if __name__ == "__main__":
    main()
