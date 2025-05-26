import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic insurance claims dataset
    """
    np.random.seed(42)
    
    # Policyholder Information
    ages = np.random.normal(40, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)  # Ensure valid ages
    
    # Young drivers (18-25) have higher risk
    young_driver = (ages <= 25).astype(int)
    
    # Vehicle Information
    vehicle_ages = np.random.normal(5, 3, n_samples).astype(int)
    vehicle_ages = np.clip(vehicle_ages, 0, 20)  # Ensure valid vehicle ages
    
    # Vehicle types (Sedan, SUV, Sports, Truck)
    vehicle_types = np.random.choice(['Sedan', 'SUV', 'Sports', 'Truck'], n_samples)
    
    # Vehicle makes (High-end vs regular)
    vehicle_makes = np.random.choice(['High-end', 'Regular'], n_samples, 
                                   p=[0.3, 0.7])
    
    # Claim Information
    past_claims = np.random.poisson(0.5, n_samples)  # Average 0.5 claims per driver
    
    # Accident types (Minor, Moderate, Severe)
    accident_types = np.random.choice(['Minor', 'Moderate', 'Severe'], n_samples,
                                    p=[0.6, 0.3, 0.1])
    
    # Regions (Urban, Suburban, Rural)
    regions = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples,
                             p=[0.4, 0.4, 0.2])
    
    # Create synthetic claim severity based on various factors
    base_severity = 5000  # Base claim amount
    
    # Factors affecting severity
    age_factor = 1 + (0.1 * young_driver)  # Young drivers have higher severity
    vehicle_age_factor = 1 + (0.05 * vehicle_ages)  # Older cars have higher severity
    
    # Vehicle type multipliers
    vehicle_type_multipliers = {
        'Sedan': 1.0,
        'SUV': 1.2,
        'Sports': 1.5,
        'Truck': 1.3
    }
    
    vehicle_type_factor = np.array([vehicle_type_multipliers[vt] for vt in vehicle_types])
    
    # Vehicle make multipliers
    vehicle_make_multipliers = {
        'High-end': 1.5,
        'Regular': 1.0
    }
    
    vehicle_make_factor = np.array([vehicle_make_multipliers[vm] for vm in vehicle_makes])
    
    # Accident type multipliers
    accident_type_multipliers = {
        'Minor': 0.5,
        'Moderate': 1.0,
        'Severe': 2.0
    }
    
    accident_type_factor = np.array([accident_type_multipliers[at] for at in accident_types])
    
    # Region multipliers
    region_multipliers = {
        'Urban': 1.2,
        'Suburban': 1.0,
        'Rural': 0.8
    }
    
    region_factor = np.array([region_multipliers[r] for r in regions])
    
    # Past claims multiplier
    past_claims_factor = 1 + (0.1 * past_claims)
    
    # Add some random noise
    noise = np.random.normal(0, 1000, n_samples)
    
    # Calculate final claim severity
    claim_severity = (base_severity * 
                     age_factor * 
                     vehicle_age_factor *
                     vehicle_type_factor *
                     vehicle_make_factor *
                     accident_type_factor *
                     region_factor *
                     past_claims_factor +
                     noise)
    
    # Ensure positive claim amounts
    claim_severity = np.maximum(claim_severity, 1000)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'policy_id': range(1, n_samples + 1),
        'driver_age': ages,
        'vehicle_age': vehicle_ages,
        'vehicle_type': vehicle_types,
        'vehicle_make': vehicle_makes,
        'past_claims': past_claims,
        'accident_type': accident_types,
        'region': regions,
        'claim_severity': claim_severity
    })
    
    return df

def save_dataset(df):
    """
    Save the dataset to CSV files
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save full dataset
    df.to_csv('data/insurance_claims.csv', index=False)
    
    # Create train/test split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"Dataset saved successfully!\n"
          f"Total samples: {len(df)}\n"
          f"Training samples: {len(train_df)}\n"
          f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    df = generate_synthetic_data(1000)
    save_dataset(df)
