"""
Carbon Footprint Estimator - Machine Learning Package
=====================================================

A comprehensive ML-based system for predicting individual carbon footprint
and providing personalized sustainability recommendations.

Modules:
    - data_generation: Generate synthetic carbon footprint dataset
    - data_preprocessing: Preprocess data for ML models
    - model_training: Train and evaluate ML models
    - suggestion_engine: Generate personalized recommendations

Example Usage:
    >>> from data_generation import generate_carbon_footprint_dataset
    >>> from data_preprocessing import preprocess_dataset
    >>> from model_training import train_and_evaluate_models
    >>> from suggestion_engine import SuggestionEngine
    >>>
    >>> # Generate dataset
    >>> df = generate_carbon_footprint_dataset(n_samples=1000)
    >>>
    >>> # Preprocess data
    >>> X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(df)
    >>>
    >>> # Train models
    >>> trainer = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    >>>
    >>> # Generate recommendations
    >>> engine = SuggestionEngine()
    >>> suggestions = engine.generate_suggestions(user_data, prediction, avg)

Author: Carbon Footprint Project Team
Version: 1.0.0
License: Open Source
Date: December 2025
"""

__version__ = "1.0.0"
__author__ = "Carbon Footprint Project Team"

# Import main classes and functions
from .data_generation import (
    generate_carbon_footprint_dataset,
    save_dataset,
    calculate_carbon_emissions
)

from .data_preprocessing import (
    DataPreprocessor,
    preprocess_dataset
)

from .model_training import (
    CarbonFootprintModel,
    ModelTrainer,
    train_and_evaluate_models
)

from .suggestion_engine import SuggestionEngine

__all__ = [
    'generate_carbon_footprint_dataset',
    'save_dataset',
    'calculate_carbon_emissions',
    'DataPreprocessor',
    'preprocess_dataset',
    'CarbonFootprintModel',
    'ModelTrainer',
    'train_and_evaluate_models',
    'SuggestionEngine'
]
