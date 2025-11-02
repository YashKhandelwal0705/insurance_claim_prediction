import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def engineer_features(df):
    """
    Create ALL features to match train_engineered.csv exactly.
    This replicates the feature_engineering.py transform() method.
    """
    df = df.copy()
    
    # 1. Create age bins
    age_bins = [18, 25, 40, 60, 80]
    age_labels = ['young', 'mid_age', 'senior', 'elderly']
    df['age_group'] = pd.cut(df['driver_age'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    # 2. Create vehicle age bins
    vehicle_age_bins = [0, 5, 10, 15, 20]
    vehicle_age_labels = ['new', 'mid_age', 'old', 'very_old']
    df['vehicle_age_group'] = pd.cut(df['vehicle_age'], bins=vehicle_age_bins, labels=vehicle_age_labels, include_lowest=True)
    
    # 3. Claims category
    df['claims_category'] = pd.cut(df['past_claims'], bins=[-1, 0, 2, 10], labels=['no_claims', 'few_claims', 'many_claims'])
    
    # 4. Risky combo (young driver + sports car)
    df['risky_combo'] = ((df['age_group'] == 'young') & (df['vehicle_type'] == 'Sports')).astype(int)
    
    # 5. Experience ratio
    df['experience_ratio'] = df['driver_age'] / (df['vehicle_age'] + 1)
    
    # 6. Claims per year (assuming started driving at 18)
    driving_years = df['driver_age'] - 18
    driving_years = driving_years.replace(0, 1)  # Avoid division by zero
    df['claims_per_year'] = df['past_claims'] / driving_years
    
    # 7. High value vehicle (high-end SUV or Sports)
    df['high_value_vehicle'] = ((df['vehicle_make'] == 'High-end') & 
                                 (df['vehicle_type'].isin(['SUV', 'Sports']))).astype(int)
    
    # 8. Urban sports combination
    df['urban_sports'] = ((df['region'] == 'Urban') & (df['vehicle_type'] == 'Sports')).astype(int)
    
    # 9. Age-accident severity interaction
    df['age_accident_severity'] = df['age_group'].astype(str) + '_' + df['accident_type'].astype(str)
    
    # 10. Vehicle profile
    df['vehicle_profile'] = df['vehicle_type'].astype(str) + '_' + df['vehicle_make'].astype(str)
    
    # 11. Driver risk score
    df['driver_risk_score'] = (df['past_claims'] * 2) + ((df['driver_age'] < 25).astype(int) * 3) + ((df['vehicle_age'] > 10).astype(int) * 1)
    
    # 12. High claim history
    df['high_claim_history'] = (df['past_claims'] > 2).astype(int)
    
    # 13. Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    num_features = ['driver_age', 'vehicle_age', 'past_claims']
    poly_features = poly.fit_transform(df[num_features])
    
    # Create feature names with 'poly_' prefix to match train_engineered.csv
    poly_feature_names = poly.get_feature_names_out(input_features=num_features)
    poly_feature_names = ['poly_' + name for name in poly_feature_names]
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df = pd.concat([df, poly_df], axis=1)
    
    # 14. Log transformations
    df['log_vehicle_age'] = np.log1p(df['vehicle_age'])
    df['log_past_claims'] = np.log1p(df['past_claims'])
    df['log_driver_age'] = np.log1p(df['driver_age'])
    
    # 15. Square root transformations
    df['sqrt_vehicle_age'] = np.sqrt(df['vehicle_age'])
    df['sqrt_driver_age'] = np.sqrt(df['driver_age'])
    
    # 16. Inverse transformations
    df['inv_vehicle_age'] = 1 / (df['vehicle_age'] + 1)
    df['inv_driver_age'] = 1 / (df['driver_age'] + 1)
    
    return df

def reorder_columns(df):
    """
    Return dataframe with columns dropped that shouldn't be in the model input.
    The model's embedded preprocessor will select the columns it needs.
    """
    # Drop the original numerical columns that were used to create polynomial features
    # But keep everything else - let the model's preprocessor handle column selection
    return df
