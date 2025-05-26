import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class BusinessTranslator:
    def __init__(self):
        """Initialize business translator"""
        self.model = None
        self.risk_threshold = 1.2  # Threshold for high risk
        
    def load_model(self):
        """Load the best trained model"""
        self.model = joblib.load('models/best_xgboost.pkl')
        return self.model
    
    def load_test_data(self):
        """Load test data"""
        test_df = pd.read_csv('data/test_engineered.csv')
        X_test = test_df.drop(['claim_severity', 'policy_id'], axis=1)
        y_test = np.log1p(test_df['claim_severity'])
        return X_test, y_test, test_df
    
    def flag_risky_policies(self, predictions, features_df):
        """Flag policies that need manual review"""
        risky_cases = []
        
        for idx, pred in enumerate(predictions):
            if pred > self.risk_threshold:
                features = features_df.iloc[idx].to_dict()
                risky_cases.append({
                    'policy_id': features.get('policy_id', f'policy_{idx}'),
                    'predicted_claim': np.expm1(pred),
                    'features': features,
                    'risk_level': 'HIGH'
                })
        
        return risky_cases
    
    def adjust_premium(self, features):
        """Calculate premium adjustment based on risk factors"""
        base_adjustment = 0.0  # Start with no adjustment
        
        # Premium increases for high-risk factors
        if features['vehicle_type'] in ['SUV', 'Luxury']:
            base_adjustment += 0.15  # 15% increase for high-end vehicles
        
        if features['driver_age'] < 25:
            base_adjustment += 0.10  # 10% increase for young drivers
        
        if features['region'] in ['Urban', 'High-Risk']:
            base_adjustment += 0.15  # 15% increase for risky regions
        
        if features['past_claims'] > 0:
            base_adjustment += 0.10  # 10% increase for previous claims
        
        return base_adjustment
    
    def generate_business_actions(self, sample_size=5):
        """Generate business actions for risky cases"""
        print("\n=== Business Actions Report ===\n")
        
        # Load model and data
        model = self.load_model()
        X_test, _, test_df = self.load_test_data()
        
        # Get predictions
        predictions = model.predict(X_test)
        
        # Flag risky policies
        risky_cases = self.flag_risky_policies(predictions, test_df)
        
        # Generate sample cases
        sample_cases = np.random.choice(risky_cases, min(sample_size, len(risky_cases)), replace=False)
        
        print("Example Business Actions:\n")
        for case in sample_cases:
            print(f"\nCase: Policy {case['policy_id']}")
            print("-" * 50)
            
            # Show key features that triggered high risk
            key_features = []
            if case['features']['vehicle_type'] in ['SUV', 'Luxury']:
                key_features.append(f"Vehicle Type: {case['features']['vehicle_type']}")
            if case['features']['driver_age'] < 25:
                key_features.append(f"Driver Age: {case['features']['driver_age']}")
            if case['features']['region'] in ['Urban', 'High-Risk']:
                key_features.append(f"Region: {case['features']['region']}")
            if case['features']['past_claims'] > 0:
                key_features.append(f"Past Claims: {case['features']['past_claims']}")
            
            print(f"Predicted Claim Severity: {case['predicted_claim']:.2f}")
            print("\nKey Risk Factors:")
            for feature in key_features:
                print(f"- {feature}")
            
            # Calculate premium adjustment
            premium_adjustment = self.adjust_premium(case['features'])
            print(f"\nPremium Adjustment: {premium_adjustment * 100:.0f}%")
            
            # Generate business actions
            print("\nRecommended Actions:")
            print("- Route claim for manual review")
            print(f"- Adjust premium by {premium_adjustment * 100:.0f}%")
            print("- Flag for additional fraud checks")
            print("- Monitor claims processing closely")
            print("\n" * 2)

def main():
    """Main function to generate business actions"""
    translator = BusinessTranslator()
    translator.generate_business_actions()

if __name__ == "__main__":
    main()
