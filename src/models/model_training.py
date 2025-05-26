import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os

class ModelTrainer:
    def __init__(self):
        """Initialize model trainer"""
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the preprocessed data"""
        train_df = pd.read_csv(r'F:\Projects\insurance_claim_prediction\data\train_engineered.csv')
        X = train_df.drop(['claim_severity', 'policy_id'], axis=1)
        y = np.log1p(train_df['claim_severity'])  # Using log-transformed target
        
        return X, y
    
    def train_linear_regression(self, X, y):
        """
        Train Linear Regression model with proper preprocessing
        """
        print("\nTraining Linear Regression...")
        
        # Define categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create preprocessing pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numerical_cols)
            ]
        )
        
        # Create pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train and evaluate
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            scores.append({
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
        
        # Save results
        self.results['Linear Regression'] = {
            'mean': {
                'MAE': np.mean([s['MAE'] for s in scores]),
                'RMSE': np.mean([s['RMSE'] for s in scores]),
                'R2': np.mean([s['R2'] for s in scores])
            },
            'std': {
                'MAE': np.std([s['MAE'] for s in scores]),
                'RMSE': np.std([s['RMSE'] for s in scores]),
                'R2': np.std([s['R2'] for s in scores])
            }
        }
        
        # Save best model
        pipe.fit(X, y)
        self.models['Linear Regression'] = pipe
        joblib.dump(pipe, r'F:\Projects\insurance_claim_prediction\models\linear_regression.pkl')
        
        print("Linear Regression training complete!")
    
    def train_random_forest(self, X, y):
        """
        Train Random Forest Regressor with GridSearchCV
        """
        print("\nTraining Random Forest Regressor...")
        
        # Define categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create preprocessing pipeline for tree-based models
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Define parameter grid
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        
        # Create pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42))
        ])
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit and evaluate
        grid_search.fit(X, y)
        
        # Save results
        self.results['Random Forest'] = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # Convert from negative MSE
            'cv_results': grid_search.cv_results_
        }
        
        # Save best model
        self.models['Random Forest'] = grid_search.best_estimator_
        joblib.dump(grid_search.best_estimator_, r'F:\Projects\insurance_claim_prediction\models\random_forest.pkl')
        
        print("Random Forest training complete!")
    
    def train_xgboost(self, X, y):
        """
        Train XGBoost Regressor with GridSearchCV
        """
        print("\nTraining XGBoost Regressor...")
        
        # Define categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create preprocessing pipeline for tree-based models
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Define parameter grid
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.3],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
        
        # Create pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(random_state=42, objective='reg:squarederror'))
        ])
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit and evaluate
        grid_search.fit(X, y)
        
        # Save results
        self.results['XGBoost'] = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # Convert from negative MSE
            'cv_results': grid_search.cv_results_
        }
        
        # Save best model
        self.models['XGBoost'] = grid_search.best_estimator_
        joblib.dump(grid_search.best_estimator_, r'F:\Projects\insurance_claim_prediction\models\xgboost.pkl')
        
        print("XGBoost training complete!")
    
    def train_all_models(self):
        """
        Train all models
        """
        X, y = self.load_data()
        
        print("\nStarting model training...")
        print("-" * 50)
        
        # Train Linear Regression
        self.train_linear_regression(X, y)
        
        # Train Random Forest
        self.train_random_forest(X, y)
        
        # Train XGBoost
        self.train_xgboost(X, y)
        
        print("\nTraining complete!")
        
        # Print summary of results
        print("\nModel Performance Summary:")
        print("-" * 50)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            if 'mean' in result:
                print(f"MAE: {result['mean']['MAE']:.4f} ± {result['std']['MAE']:.4f}")
                print(f"RMSE: {result['mean']['RMSE']:.4f} ± {result['std']['RMSE']:.4f}")
                print(f"R2: {result['mean']['R2']:.4f} ± {result['std']['R2']:.4f}")
            else:
                print(f"Best CV Score: {result['best_score']:.4f}")
                print(f"Best Parameters: {result['best_params']}")

def main():
    """Main function to train all models"""
    trainer = ModelTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main()
