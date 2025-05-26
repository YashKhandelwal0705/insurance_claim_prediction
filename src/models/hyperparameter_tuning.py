import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    def __init__(self):
        """Initialize hyperparameter tuner"""
        self.best_models = {}
        self.results = {}
    
    def load_data(self):
        """Load the preprocessed data"""
        train_df = pd.read_csv('data/train_engineered.csv')
        X = train_df.drop(['claim_severity', 'policy_id'], axis=1)
        y = np.log1p(train_df['claim_severity'])
        
        return X, y
    
    def get_categorical_numerical_cols(self, X):
        """Get categorical and numerical columns"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        return categorical_cols, numerical_cols
    
    def create_preprocessor(self, categorical_cols):
        """Create preprocessing pipeline for tree-based models"""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        return preprocessor
    
    def tune_random_forest(self, X, y):
        """
        Tune Random Forest Regressor using RandomizedSearchCV
        """
        print("\nTuning Random Forest Regressor...")
        
        # Get column types
        categorical_cols, _ = self.get_categorical_numerical_cols(X)
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessor(categorical_cols)
        
        # Define parameter distributions
        param_dist = {
            'model__n_estimators': randint(100, 500),
            'model__max_depth': randint(5, 30),
            'model__min_samples_split': randint(2, 15),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', None],
            'model__bootstrap': [True, False]
        }
        
        # Create pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42))
        ])
        
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=100,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # Fit and evaluate
        random_search.fit(X, y)
        
        # Save results
        self.results['Random Forest'] = {
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
        
        # Save best model
        self.best_models['Random Forest'] = random_search.best_estimator_
        joblib.dump(random_search.best_estimator_, 'models/best_random_forest.pkl')
        
        print("\nRandom Forest tuning complete!")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {-random_search.best_score_:.4f}")
    
    def tune_xgboost(self, X, y):
        """
        Tune XGBoost Regressor using RandomizedSearchCV
        """
        print("\nTuning XGBoost Regressor...")
        
        # Get column types
        categorical_cols, _ = self.get_categorical_numerical_cols(X)
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessor(categorical_cols)
        
        # Define parameter distributions
        param_dist = {
            'model__n_estimators': randint(100, 500),
            'model__max_depth': randint(3, 10),
            'model__learning_rate': uniform(0.01, 0.3),
            'model__subsample': uniform(0.7, 0.3),
            'model__colsample_bytree': uniform(0.7, 0.3),
            'model__gamma': uniform(0, 1),
            'model__reg_alpha': uniform(0, 1),
            'model__reg_lambda': uniform(0, 1),
            'model__min_child_weight': randint(1, 10)
        }
        
        # Create pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(
                random_state=42,
                objective='reg:squarederror',
                tree_method='hist'  # Faster for large datasets
            ))
        ])
        
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=100,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # Fit and evaluate
        random_search.fit(X, y)
        
        # Save results
        self.results['XGBoost'] = {
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
        
        # Save best model
        self.best_models['XGBoost'] = random_search.best_estimator_
        joblib.dump(random_search.best_estimator_, 'models/best_xgboost.pkl')
        
        print("\nXGBoost tuning complete!")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {-random_search.best_score_:.4f}")
    
    def tune_all_models(self):
        """
        Tune all models
        """
        X, y = self.load_data()
        
        print("\nStarting hyperparameter tuning...")
        print("-" * 50)
        
        # Tune Random Forest
        self.tune_random_forest(X, y)
        
        # Tune XGBoost
        self.tune_xgboost(X, y)
        
        print("\nTuning complete!")

def main():
    """Main function to tune all models"""
    tuner = HyperparameterTuner()
    tuner.tune_all_models()

if __name__ == "__main__":
    main()
