import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.inspection import permutation_importance

class ModelExplainability:
    def __init__(self):
        """Initialize model explainability"""
        self.shap_values = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load test data"""
        test_df = pd.read_csv('data/test_engineered.csv')
        X_test = test_df.drop(['claim_severity', 'policy_id'], axis=1)
        y_test = np.log1p(test_df['claim_severity'])
        return X_test, y_test
    
    def load_model(self, model_path):
        """Load trained model"""
        return joblib.load(model_path)
    
    def get_feature_importance(self, model, X_test):
        """Get feature importance for tree-based models"""
        # Get feature names from preprocessor
        preprocessor = model.named_steps['preprocessor']
        
        # Get categorical feature names
        cat_features = preprocessor.transformers_[0][1].get_feature_names_out()
        
        # Get numerical feature names
        num_features = X_test.select_dtypes(include=['int64', 'float64']).columns
        
        # Combine all feature names
        feature_names = list(cat_features) + list(num_features)
        
        # Get feature importance
        feature_importance = model.named_steps['model'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        
        # Group importance by original feature names
        importance_df['original_feature'] = importance_df['feature'].str.split('_', expand=True)[0]
        grouped_importance = importance_df.groupby('original_feature')['importance'].sum().reset_index()
        
        return grouped_importance.sort_values('importance', ascending=False)
    
    def calculate_shap_values(self, model, X_test, model_name):
        """Calculate SHAP values"""
        # Get preprocessor and model
        preprocessor = model.named_steps['preprocessor']
        model = model.named_steps['model']
        
        # Transform data
        X_transformed = preprocessor.transform(X_test)
        
        # Convert to numpy array for SHAP
        X_array = X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed
        
        # Create SHAP Explainer
        explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        shap_values = explainer(X_array)
        
        # Get feature names
        cat_features = preprocessor.transformers_[0][1].get_feature_names_out()
        num_features = X_test.select_dtypes(include=['int64', 'float64']).columns
        feature_names = list(cat_features) + list(num_features)
        
        # Store results
        self.shap_values[model_name] = {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_test': X_test,
            'X_transformed': X_transformed,
            'feature_names': feature_names
        }
        
        return shap_values
    
    def plot_feature_importance(self, importance_df, model_name):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x='importance',
            y='original_feature',
            data=importance_df.head(15)
        )
        plt.title(f'Top 15 Feature Importance for {model_name}')
        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_name}_feature_importance.png')
        plt.close()
    
    def plot_shap_summary(self, model_name):
        """Plot SHAP summary plot"""
        shap_values = self.shap_values[model_name]['shap_values']
        feature_names = self.shap_values[model_name]['feature_names']
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            feature_names,
            plot_type='bar',
            max_display=15,
            show=False
        )
        plt.title(f'SHAP Feature Importance for {model_name}')
        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_name}_shap_summary.png')
        plt.close()
    
    def plot_shap_waterfall(self, model_name, instance_idx=0):
        """Plot SHAP waterfall plot for a single instance"""
        shap_values = self.shap_values[model_name]['shap_values']
        X_transformed = self.shap_values[model_name]['X_transformed']
        
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(
            shap_values[instance_idx],
            max_display=15,
            show=False
        )
        plt.title(f'SHAP Waterfall Plot for {model_name} - Instance {instance_idx}')
        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_name}_shap_waterfall_{instance_idx}.png')
        plt.close()
    
    def analyze_model(self, model_path, model_name):
        """Analyze a single model"""
        print(f"\nAnalyzing {model_name}...")
        
        # Load model and data
        model = self.load_model(model_path)
        X_test, y_test = self.load_data()
        
        # Get feature importance
        importance_df = self.get_feature_importance(model, X_test)
        self.feature_importance[model_name] = importance_df
        
        # Plot feature importance
        self.plot_feature_importance(importance_df, model_name)
        
        # Calculate and plot SHAP values
        shap_values = self.calculate_shap_values(model, X_test, model_name)
        self.plot_shap_summary(model_name)
        self.plot_shap_waterfall(model_name)
        
        print(f"\nTop 5 features affecting {model_name} predictions:")
        print(importance_df.head(5))
        
        # Generate business insights
        self.generate_business_insights(importance_df, model_name)
    
    def generate_business_insights(self, importance_df, model_name):
        """Generate business insights from feature importance"""
        print(f"\nBusiness Insights for {model_name}:")
        
        # Get top features
        top_features = importance_df.head(5)['original_feature'].tolist()
        
        # Map features to business insights
        insights = {
            'accident': 'Accident type and severity significantly impact claim costs',
            'vehicle': 'Vehicle characteristics (type, make, age) are key risk factors',
            'region': 'Geographic location affects claim severity',
            'driver': 'Driver demographics and behavior influence risk',
            'age': 'Driver and vehicle age are important risk factors'
        }
        
        print("\nKey Risk Factors:")
        for feature in top_features:
            if feature in insights:
                print(f"- {feature}: {insights[feature]}")
        
        # Generate more specific insights based on feature importance
        print("\nDetailed Business Insights:")
        
        # Analyze accident feature
        if 'accident' in top_features:
            print("- Accident type and severity are the most significant predictors of claim costs")
            print("  * More severe accidents lead to higher claim amounts")
            print("  * Accident type (collision vs non-collision) affects repair costs")
        
        # Analyze vehicle features
        if 'vehicle' in top_features:
            print("- Vehicle characteristics significantly impact claim severity:")
            print("  * Luxury and high-end vehicles have higher repair costs")
            print("  * SUVs and trucks may have higher claim costs due to size")
            print("  * Newer vehicles typically have higher repair costs")
        
        # Analyze region features
        if 'region' in top_features:
            print("- Geographic location affects claim severity:")
            print("  * Urban areas may have higher claim frequencies")
            print("  * Certain regions may have higher repair costs")
            print("  * Weather patterns in different regions affect claim types")
        
        # Analyze driver features
        if 'driver' in top_features:
            print("- Driver characteristics influence claim severity:")
            print("  * Young drivers may have higher risk profiles")
            print("  * Driving history and experience affect claim likelihood")
            print("  * Demographics (age, gender) influence risk levels")
        
        # Analyze age features
        if 'age' in top_features:
            print("- Age factors are important predictors:")
            print("  * Both driver and vehicle age affect claim severity")
            print("  * Older vehicles may have higher repair costs")
            print("  * Younger drivers typically have higher claim rates")
    
    def analyze_all_models(self):
        """Analyze all models"""
        print("\nStarting model explainability analysis...")
        print("-" * 50)
        
        # Analyze Random Forest
        self.analyze_model('models/best_random_forest.pkl', 'Random Forest')
        
        # Analyze XGBoost
        self.analyze_model('models/best_xgboost.pkl', 'XGBoost')
        
        print("\nAnalysis complete!")

def main():
    """Main function to analyze all models"""
    analyzer = ModelExplainability()
    analyzer.analyze_all_models()

if __name__ == "__main__":
    main()
