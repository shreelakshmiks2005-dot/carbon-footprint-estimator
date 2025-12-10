"""
Data Preprocessing Module
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

class DataPreprocessor:
    """Preprocess data for model training"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_cols = None
    
    def fit(self, df):
        """Fit preprocessor on training data"""
        from sklearn.preprocessing import StandardScaler
        
        df = df.copy()
        
        # Identify and encode only categorical variables (object dtype)
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale all numeric features
        self.scaler = StandardScaler()
        self.scaler.fit(df)
        self.feature_columns = df.columns.tolist()
        
        return self
    
    def transform(self, df):
        """Transform data - handles unseen categories"""
        from sklearn.preprocessing import StandardScaler
        
        df = df.copy()
        # Ensure we have the feature columns seen during fit.
        # Add missing columns with default values (0) and drop any extra columns.
        if self.feature_columns is None:
            raise ValueError("Preprocessor has no feature_columns set. Did you fit it?")

        # Add missing columns with zeros
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Drop extra columns that were not present during fit
        extra_cols = [c for c in df.columns if c not in self.feature_columns]
        if extra_cols:
            df = df.drop(columns=extra_cols)

        # Reorder columns to match training order
        df = df[self.feature_columns]

        # Encode categorical columns - handle unseen values
        for col in (self.categorical_cols or []):
            if col in df.columns:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str)
                known_mask = df[col].isin(le.classes_)
                if known_mask.any():
                    df.loc[known_mask, col] = le.transform(df.loc[known_mask, col])
                # For unseen values, set to first class index (0)
                df.loc[~known_mask, col] = 0

        # Scale features (StandardScaler expects same column order)
        return self.scaler.transform(df)
    
    def fit_transform(self, df):
        """Fit and transform"""
        from sklearn.preprocessing import StandardScaler
        
        df = df.copy()
        
        # Identify categorical columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical variables
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale features
        self.scaler = StandardScaler()
        self.scaler.fit(df)
        self.feature_columns = df.columns.tolist()
        
        return self.scaler.transform(df)
    
    def save(self, filepath):
        """Save preprocessor"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath):
        """Load preprocessor"""
        return joblib.load(filepath)
