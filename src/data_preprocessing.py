"""
Data Preprocessing Module - Handles data cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class DataPreprocessor:
    """Preprocesses carbon footprint data for ML models"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.cat_modes = {}
        self.feature_names = None
        self.categorical_columns = [
            'transportation_mode', 'diet_type', 'car_type', 'heating_fuel_type'
        ]
        self.numeric_columns = [
            'commute_distance_km', 'flights_per_year', 'meat_consumption_kg_per_month',
            'electricity_usage_kwh', 'gas_usage_therms', 'water_usage_gallons',
            'recycling_percentage', 'new_clothes_per_month',
            'household_size', 'vehicle_fuel_efficiency_km_per_l', 'renewable_energy_percentage'
        ]
    
    def fit_transform(self, df):
        """Fit preprocessor on data and transform it"""
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        # Encode categorical variables
        df = self._encode_categorical(df, fit=True)
        
        # Normalize numeric features
        df = self._normalize_features(df, fit=True)
        
        return df
    
    def transform(self, df):
        """Transform data using fitted preprocessor"""
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        # Encode categorical variables
        df = self._encode_categorical(df, fit=False)
        
        # Normalize numeric features
        df = self._normalize_features(df, fit=False)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Convert recycling_percentage to numeric if it contains list strings like "['Metal']"
        if 'recycling_percentage' in df.columns:
            df['recycling_percentage'] = df['recycling_percentage'].astype(str).apply(
                lambda x: 50.0 if (x.startswith('[') and x.endswith(']')) else float(x) if x.replace('.', '').replace('-', '').isdigit() else 0.0
            )
        
        # Fill numeric columns with median
        for col in self.numeric_columns:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
        
        return df
    
    def _feature_engineering(self, df):
        """Create new features from existing ones"""
        df = df.copy()
        
        # Total transport emissions category
        df['high_transport_user'] = (
            (df['commute_distance_km'] > df['commute_distance_km'].median()).astype(int) |
            (df['flights_per_year'] > 2).astype(int)
        ).astype(int)
        
        # Energy consumption category
        df['high_energy_user'] = (
            (df['electricity_usage_kwh'] + df['gas_usage_therms'] * 2.93) > 
            (df['electricity_usage_kwh'].median() + df['gas_usage_therms'].median() * 2.93)
        ).astype(int)
        
        # Sustainability score (0-100)
        df['sustainability_score'] = (
            df['recycling_percentage'] * 0.5 +
            (100 - df['new_clothes_per_month'] * 10).clip(lower=0) * 0.5
        )
        
        # Total water and waste factor
        df['waste_water_factor'] = df['water_usage_gallons'] + df['new_clothes_per_month'] * 100
        
        return df
    
    def _encode_categorical(self, df, fit=False):
        """Encode categorical variables"""
        df = df.copy()
        
        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    # Save the mode for the column so we can map unseen labels at transform time
                    try:
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            self.cat_modes[col] = mode_val[0]
                        else:
                            self.cat_modes[col] = None
                    except Exception:
                        self.cat_modes[col] = None

                    self.label_encoders[col] = LabelEncoder()
                    # Fit encoder on the training column
                    self.label_encoders[col].fit(df[col].astype(str))
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    # Replace unseen labels with the training-mode (or a known class) to avoid transform errors
                    le = self.label_encoders.get(col)
                    if le is None:
                        continue

                    known = set(le.classes_)

                    def _map_value(v):
                        try:
                            # Keep as string for comparison since encoders were fit on strings
                            vs = str(v)
                        except Exception:
                            vs = v

                        if vs in known:
                            return vs
                        # fallback to saved mode, or first known class
                        fallback = self.cat_modes.get(col)
                        if fallback is None:
                            fallback = le.classes_[0]
                        return str(fallback)

                    df[col] = df[col].apply(_map_value).astype(str)
                    df[col] = le.transform(df[col])
        
        return df
    
    def _normalize_features(self, df, fit=False):
        """Normalize numeric features using StandardScaler"""
        df = df.copy()
        
        numeric_cols = [col for col in self.numeric_columns if col in df.columns]
        numeric_cols.extend(['high_transport_user', 'high_energy_user', 
                           'sustainability_score', 'waste_water_factor'])
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        self.feature_names = numeric_cols
        
        return df
    
    def split_data(self, df):
        """Split data into train and test sets"""
        X = df.drop('monthly_carbon_footprint_kg', axis=1)
        y = df['monthly_carbon_footprint_kg']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def save(self, path='models/preprocessor.pkl'):
        """Save preprocessor to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path='models/preprocessor.pkl'):
        """Load preprocessor from file"""
        return joblib.load(path)


def preprocess_dataset(df, test_size=0.2, random_state=42):
    """
    Main preprocessing function
    
    Args:
        df: Input DataFrame
        test_size: Test set size
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    preprocessor = DataPreprocessor(test_size=test_size, random_state=random_state)
    
    # Fit and transform
    df_processed = preprocessor.fit_transform(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == '__main__':
    from data_generation import generate_carbon_footprint_dataset
    
    # Generate and preprocess data
    df = generate_carbon_footprint_dataset(n_samples=1000)
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(df)
    
    print("Data preprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nFeature names: {preprocessor.feature_names}")
    
    # Save preprocessor
    preprocessor.save()
