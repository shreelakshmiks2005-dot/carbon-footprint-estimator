"""
Data Enrichment Module - Adds derived features to Kaggle dataset
Creates household size, vehicle fuel efficiency, renewable energy %, heating type
based on existing data patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def enrich_dataset(df):
    """
    Enrich dataset by deriving additional lifestyle features from existing data.
    
    Args:
        df: DataFrame with original Kaggle dataset columns
    
    Returns:
        Enriched DataFrame with additional household/energy columns
    """
    df = df.copy()
    
    # 1. Derive household_size from commute distance and carbon footprint patterns
    # Assumption: higher carbon footprint with moderate distance = larger household
    df['household_size'] = derive_household_size(df)
    
    # 2. Derive vehicle_fuel_efficiency_km_per_l from transportation mode and distance
    # Electric vehicles get higher efficiency, petrol/diesel get standard/low
    df['vehicle_fuel_efficiency_km_per_l'] = derive_vehicle_efficiency(df)
    
    # 3. Derive renewable_energy_percentage from existing data patterns
    # Assumption: higher recycling % correlates with renewable energy adoption
    df['renewable_energy_percentage'] = derive_renewable_percentage(df)
    
    # 4. Derive heating_fuel_type from gas usage patterns
    df['heating_fuel_type'] = derive_heating_fuel_type(df)
    
    return df


def derive_household_size(df):
    """
    Estimate household size (1-6 people) based on:
    - Carbon footprint level
    - Commute distance
    - New clothes purchases
    """
    carbon = df['monthly_carbon_footprint_kg'].astype(float)
    distance = df['commute_distance_km'].astype(float)
    clothes = df['new_clothes_per_month'].astype(float)
    
    # Normalize each component (0-1)
    carbon_norm = (carbon - carbon.min()) / (carbon.max() - carbon.min() + 1)
    distance_norm = (distance - distance.min()) / (distance.max() - distance.min() + 1)
    clothes_norm = (clothes - clothes.min()) / (clothes.max() - clothes.min() + 1)
    
    # Weighted composite: carbon footprint is strongest indicator
    score = 0.6 * carbon_norm + 0.2 * distance_norm + 0.2 * clothes_norm
    
    # Map to household size (1-6)
    household_size = (score * 5 + 1).round().astype(int)
    household_size = household_size.clip(lower=1, upper=6)
    
    return household_size


def derive_vehicle_efficiency(df):
    """
    Estimate vehicle fuel efficiency (km/l) based on:
    - Vehicle type (electric, hybrid, petrol, diesel)
    - Commute distance (proxy for driving habits)
    """
    car_type = df['car_type'].fillna('petrol').str.lower()
    distance = df['commute_distance_km'].astype(float)
    
    # Base efficiency by vehicle type
    efficiency = car_type.map({
        'electric': 25.0,  # km/l equivalent (very efficient)
        'hybrid': 18.0,    # hybrid average
        'petrol': 12.0,    # petrol average
        'diesel': 14.0,    # diesel average
        'nan': 12.0        # default
    }).fillna(12.0)
    
    # Adjust based on distance (longer distances = less efficient driving)
    distance_norm = (distance - distance.min()) / (distance.max() - distance.min() + 1)
    efficiency_adjustment = 1 - (distance_norm * 0.3)  # -30% variation
    efficiency = efficiency * efficiency_adjustment
    
    # Add small random noise for realism
    noise = np.random.normal(1.0, 0.05, len(df))
    efficiency = (efficiency * noise).round(1)
    efficiency = efficiency.clip(lower=5.0, upper=30.0)
    
    return efficiency


def derive_renewable_percentage(df):
    """
    Estimate renewable energy percentage (0-100%) based on:
    - Recycling percentage (proxy for environmental consciousness)
    - Low carbon footprint (indicator of sustainable choices)
    """
    recycling = df['recycling_percentage'].fillna('0%').astype(str)
    
    # Parse recycling percentage (may be list like "['Metal']" or string like "50%")
    # Convert list strings to indicator (50% if recycling, 0% if not)
    def parse_recycling(val):
        val = str(val).strip()
        if val.startswith('[') and val.endswith(']'):
            # It's a list like "['Metal']" - treat as recycling = 50%
            return 50.0
        elif '%' in val:
            # It's a percentage string
            return float(val.rstrip('%'))
        else:
            # Try to parse as number
            try:
                return float(val)
            except:
                return 0.0
    
    recycling_pct = recycling.apply(parse_recycling)
    
    carbon = df['monthly_carbon_footprint_kg'].astype(float)
    
    # Normalize
    recycling_norm = recycling_pct / 100.0
    carbon_norm = 1 - ((carbon - carbon.min()) / (carbon.max() - carbon.min() + 1))  # inverse
    
    # Weighted composite
    renewable_pct = (0.7 * recycling_norm + 0.3 * carbon_norm) * 100
    
    # Add some variation (not everyone with high recycling has solar)
    noise = np.random.normal(1.0, 0.15, len(df))
    renewable_pct = (renewable_pct * noise).round(0)
    renewable_pct = renewable_pct.clip(lower=0, upper=100).astype(int)
    
    return renewable_pct


def derive_heating_fuel_type(df):
    """
    Estimate primary heating fuel based on:
    - Gas usage (if high, likely gas heating)
    - Transportation mode and carbon footprint (urban vs rural, wealthy vs not)
    """
    gas_usage = df['gas_usage_therms'].astype(float)
    transport = df['transportation_mode'].fillna('private').str.lower()
    
    # Gas usage as primary signal
    gas_norm = (gas_usage - gas_usage.min()) / (gas_usage.max() - gas_usage.min() + 1)
    
    # Transportation as secondary signal (public transport = urban = may have electric heating)
    is_public_transport = transport.isin(['public', 'walk/bicycle']).astype(float)
    
    # Decision logic
    heating_type = np.where(
        gas_norm > 0.6,
        'Gas',
        np.where(
            is_public_transport > 0.5,
            'Electric',
            np.where(
                gas_norm > 0.3,
                'Gas',
                np.random.choice(['Gas', 'Electric', 'Oil', 'None'], 1)[0]
            )
        )
    )
    
    return heating_type


def save_enriched_dataset(df, output_path='data/carbon_footprint_data_enriched.csv'):
    """Save enriched dataset"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Enriched dataset saved to {output_path}")
    return output_path


if __name__ == '__main__':
    # Load original dataset
    df = pd.read_csv('data/carbon_footprint_data.csv')
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Enrich
    df_enriched = enrich_dataset(df)
    print(f"\nEnriched dataset shape: {df_enriched.shape}")
    print(f"New columns added: {df_enriched.columns.difference(df.columns).tolist()}")
    
    # Show sample
    print("\nSample enriched data:")
    print(df_enriched[['transportation_mode', 'commute_distance_km', 'monthly_carbon_footprint_kg',
                       'household_size', 'vehicle_fuel_efficiency_km_per_l',
                       'renewable_energy_percentage', 'heating_fuel_type']].head(5))
    
    # Save
    save_enriched_dataset(df_enriched)
