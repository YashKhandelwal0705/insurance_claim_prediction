import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class FeatureEngineering:
    def __init__(self):
        """Initialize feature engineering class"""
        self.age_bins = [18, 25, 40, 60, 80]
        self.vehicle_age_bins = [0, 5, 10, 15, 20]
        
    def create_bins(self, df):
        """
        Create meaningful bins for numerical features
        """
        # Driver age bins
        df['age_group'] = pd.cut(df['driver_age'], bins=self.age_bins,
                                labels=['young', 'mid_age', 'senior', 'elderly'])
        
        # Vehicle age bins
        df['vehicle_age_group'] = pd.cut(df['vehicle_age'], bins=self.vehicle_age_bins,
                                       labels=['new', 'mid_age', 'old', 'very_old'])
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features based on EDA insights
        """
        # Interaction between age group and vehicle type
        df['age_vehicle_type'] = df['age_group'].astype(str) + '_' + df['vehicle_type'].astype(str)
        
        # Interaction between vehicle age and make
        df['vehicle_age_make'] = df['vehicle_age_group'].astype(str) + '_' + df['vehicle_make'].astype(str)
        
        # Interaction between age group and vehicle make
        df['age_make'] = df['age_group'].astype(str) + '_' + df['vehicle_make'].astype(str)
        
        return df
    
    def create_polynomial_features(self, df):
        """
        Create polynomial features for numerical variables
        Based on EDA insights, we'll create polynomial features for:
        - driver_age
        - vehicle_age
        - past_claims
        """
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        
        # Select numerical features for polynomial transformation
        num_features = ['driver_age', 'vehicle_age', 'past_claims']
        
        # Transform numerical features
        num_df = df[num_features]
        poly_features = poly.fit_transform(num_df)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(input_features=num_features)
        
        # Create DataFrame of polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Add polynomial features to original DataFrame
        df = pd.concat([df, poly_df], axis=1)
        
        return df
    
    def create_feature_pipeline(self):
        """
        Create preprocessing pipelines for different model types
        """
        # Define categorical columns
        categorical_cols = [
            'vehicle_type', 'vehicle_make', 'accident_type', 'region',
            'age_group', 'vehicle_age_group',
            'age_vehicle_type', 'vehicle_age_make', 'age_make'
        ]
        
        # Define numerical columns
        numerical_cols = [
            'driver_age', 'vehicle_age', 'past_claims',
            'driver_age^2', 'vehicle_age^2', 'past_claims^2',
            'driver_age vehicle_age', 'driver_age past_claims',
            'vehicle_age past_claims'
        ]
        
        # Create preprocessing pipelines
        # For tree-based models
        tree_preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', 'passthrough', numerical_cols)
            ]
        )
        
        # For linear models
        linear_preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numerical_cols)
            ]
        )
        
        return tree_preprocessor, linear_preprocessor
    
    def transform(self, df):
        """
        Apply all feature engineering steps
        """
        # Create bins
        df = self.create_bins(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create polynomial features
        df = self.create_polynomial_features(df)
        
        return df

def main():
    """Main function to execute feature engineering pipeline"""
    # Load data
    train_df = pd.read_csv(r'F:\Projects\insurance_claim_prediction\data\train.csv')
    test_df = pd.read_csv(r'F:\Projects\insurance_claim_prediction\data\test.csv')
    
    # Initialize feature engineering
    fe = FeatureEngineering()
    
    # Transform training and test data
    train_df = fe.transform(train_df)
    test_df = fe.transform(test_df)
    
    # Create preprocessing pipelines
    tree_preprocessor, linear_preprocessor = fe.create_feature_pipeline()
    
    # Save preprocessing objects
    os.makedirs(r'F:\Projects\insurance_claim_prediction\models', exist_ok=True)
    joblib.dump(tree_preprocessor, r'F:\Projects\insurance_claim_prediction\models\tree_preprocessor.pkl')
    joblib.dump(linear_preprocessor, r'F:\Projects\insurance_claim_prediction\models\linear_preprocessor.pkl')
    
    print("Feature engineering completed!")
    print(f"Training features: {train_df.shape[1]}")
    print(f"Test features: {test_df.shape[1]}")
    
    # Save transformed data
    train_df.to_csv(r'F:\Projects\insurance_claim_prediction\data\train_engineered.csv', index=False)
    test_df.to_csv(r'F:\Projects\insurance_claim_prediction\data\test_engineered.csv', index=False)

if __name__ == "__main__":
    main()
