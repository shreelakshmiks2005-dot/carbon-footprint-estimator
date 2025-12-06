"""
Data Generation Module - Creates synthetic carbon footprint dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_carbon_footprint_dataset(n_samples=1000, random_state=42):
    """
    Generate synthetic carbon footprint dataset with lifestyle features
    
    Features:
    - transportation_mode: Type of commute (Car, Public Transport, Bike, Walk)
    - commute_distance_km: Daily commute distance
    - flights_per_year: Number of flights per year
    - diet_type: Diet type (Vegan, Vegetarian, Meat-based)
    - meat_consumption_kg_per_month: Meat consumption in kg
    - electricity_usage_kwh: Monthly electricity usage
    - gas_usage_therms: Monthly natural gas usage
    - water_usage_gallons: Monthly water usage
    - recycling_percentage: Percentage of waste recycled
    - new_clothes_per_month: New clothes purchased per month
    - car_type: Type of car (Electric, Hybrid, Petrol, Diesel, None)
    
    Target:
    - monthly_carbon_footprint_kg: Total monthly CO2 emissions in kg
    """
    
    np.random.seed(random_state)
    
    data = {
        'transportation_mode': np.random.choice(
            ['Car', 'Public Transport', 'Bike', 'Walk'], 
            n_samples, 
            p=[0.4, 0.35, 0.15, 0.1]
        ),
        'commute_distance_km': np.random.uniform(0, 50, n_samples),
        'flights_per_year': np.random.poisson(2, n_samples),
        'diet_type': np.random.choice(
            ['Vegan', 'Vegetarian', 'Meat-based'], 
            n_samples,
            p=[0.1, 0.25, 0.65]
        ),
        'meat_consumption_kg_per_month': np.random.uniform(0, 20, n_samples),
        'electricity_usage_kwh': np.random.uniform(100, 1000, n_samples),
        'gas_usage_therms': np.random.uniform(0, 200, n_samples),
        'water_usage_gallons': np.random.uniform(500, 5000, n_samples),
        'recycling_percentage': np.random.uniform(0, 100, n_samples),
        'new_clothes_per_month': np.random.uniform(0, 10, n_samples),
        'car_type': np.random.choice(
            ['Electric', 'Hybrid', 'Petrol', 'Diesel', 'None'], 
            n_samples,
            p=[0.05, 0.1, 0.35, 0.2, 0.3]
        )
    }
    
    df = pd.DataFrame(data)
    
    # Calculate carbon footprint based on features
    carbon_footprint = calculate_carbon_emissions(df)
    df['monthly_carbon_footprint_kg'] = carbon_footprint
    
    return df


def calculate_carbon_emissions(df):
    """
    Calculate monthly carbon footprint based on lifestyle factors
    
    Emission factors (approximate):
    - Car: 0.21 kg CO2/km
    - Public Transport: 0.05 kg CO2/km
    - Bike/Walk: 0 kg CO2/km
    - Flight: 0.255 kg CO2/km (4 km per person per 5-hour flight)
    - Electricity: 0.92 kg CO2/kWh (varies by region)
    - Natural Gas: 5.3 kg CO2/therm
    - Water: 0.35 kg CO2/gallon
    - Meat: 27 kg CO2/kg
    - Vegetarian: 12 kg CO2/kg equivalent
    - Vegan: 2 kg CO2/kg equivalent
    - Clothes: 6.5 kg CO2/new item
    """
    
    emissions = pd.Series(0.0, index=df.index)
    
    # Transportation emissions
    transport_emissions = df.apply(lambda row: calculate_transport_emissions(row), axis=1)
    emissions += transport_emissions
    
    # Flight emissions (assuming 1000 km average per flight, 20 working days per month)
    emissions += df['flights_per_year'] * 1000 * 0.255 / 12
    
    # Diet emissions
    diet_emissions = df.apply(lambda row: calculate_diet_emissions(row), axis=1)
    emissions += diet_emissions
    
    # Electricity emissions
    emissions += df['electricity_usage_kwh'] * 0.92
    
    # Gas emissions
    emissions += df['gas_usage_therms'] * 5.3
    
    # Water emissions
    emissions += df['water_usage_gallons'] * 0.35
    
    # Recycling offset (reduces emissions)
    emissions -= df['recycling_percentage'] * 0.1
    emissions = emissions.clip(lower=0)
    
    # Clothing emissions
    emissions += df['new_clothes_per_month'] * 6.5
    
    # Add some random noise
    emissions += np.random.normal(0, 2, len(df))
    emissions = emissions.clip(lower=0)
    
    return emissions


def calculate_transport_emissions(row):
    """Calculate transport-related emissions"""
    transport_mode = row['transportation_mode']
    distance = row['commute_distance_km']
    
    # 20 working days per month
    monthly_distance = distance * 20
    
    if transport_mode == 'Car':
        # Check car type for emission factor
        car_type = row['car_type']
        if car_type == 'Electric':
            factor = 0.05
        elif car_type == 'Hybrid':
            factor = 0.10
        elif car_type == 'Petrol':
            factor = 0.21
        elif car_type == 'Diesel':
            factor = 0.18
        else:
            factor = 0
        return monthly_distance * factor
    elif transport_mode == 'Public Transport':
        return monthly_distance * 0.05
    elif transport_mode in ['Bike', 'Walk']:
        return 0
    return 0


def calculate_diet_emissions(row):
    """Calculate diet-related emissions"""
    diet_type = row['diet_type']
    meat_consumption = row['meat_consumption_kg_per_month']
    
    if diet_type == 'Vegan':
        return 2
    elif diet_type == 'Vegetarian':
        return 12
    else:  # Meat-based
        return meat_consumption * 27 + 12


def save_dataset(df, output_path='data/carbon_footprint_data.csv'):
    """Save dataset to CSV file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    return output_path


if __name__ == '__main__':
    # Generate and save dataset
    df = generate_carbon_footprint_dataset(n_samples=1000)
    save_dataset(df)
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())
