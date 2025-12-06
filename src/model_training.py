"""
Model Training Module - Builds and evaluates ML models for carbon footprint prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import json


class CarbonFootprintModel:
    """Wrapper for carbon footprint prediction models"""
    
    def __init__(self, model_type='linear'):
        """
        Initialize model
        
        Args:
            model_type: 'linear', 'random_forest', or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        self._build_model()
    
    def _build_model(self):
        """Build the selected model"""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        print(f"{self.model_type} model trained successfully!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return self.metrics, y_pred
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance (for tree-based models)"""
        if self.model_type in ['random_forest', 'xgboost']:
            importances = self.model.feature_importances_
            
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(self.model.coef_))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': np.abs(self.model.coef_)
            }).sort_values('coefficient', ascending=False)
            
            return importance_df
    
    def save(self, path):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load(path):
        """Load model from file"""
        return joblib.load(path)


class ModelTrainer:
    """Trains and compares multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_all_models(self, X_train, y_train):
        """Train all three models"""
        model_types = ['linear', 'random_forest', 'xgboost']
        
        for model_type in model_types:
            print(f"\nTraining {model_type} model...")
            model = CarbonFootprintModel(model_type=model_type)
            model.train(X_train, y_train)
            self.models[model_type] = model
        
        print("\nAll models trained successfully!")
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models"""
        for model_type, model in self.models.items():
            print(f"\nEvaluating {model_type} model...")
            metrics, y_pred = model.evaluate(X_test, y_test)
            self.results[model_type] = {
                'metrics': metrics,
                'predictions': y_pred
            }
            
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  R²: {metrics['R2']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    def get_best_model(self):
        """Get the best performing model based on R²"""
        best_model = max(
            self.results.items(),
            key=lambda x: x[1]['metrics']['R2']
        )
        return best_model[0], best_model[1]['metrics']['R2']
    
    def get_comparison_results(self):
        """Get comparison results as DataFrame"""
        comparison_data = []
        
        for model_type, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_type.replace('_', ' ').title(),
                'MAE': f"{metrics['MAE']:.4f}",
                'RMSE': f"{metrics['RMSE']:.4f}",
                'R²': f"{metrics['R2']:.4f}",
                'MAPE': f"{metrics['MAPE']:.2f}%"
            })
        
        return pd.DataFrame(comparison_data)
    
    def save_all_models(self, model_dir='models'):
        """Save all trained models"""
        for model_type, model in self.models.items():
            path = f"{model_dir}/{model_type}_model.pkl"
            model.save(path)
    
    def save_results(self, results_path='models/model_results.json'):
        """Save evaluation results to JSON"""
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        results_to_save = {}
        for model_type, result in self.results.items():
            results_to_save[model_type] = {
                'metrics': {k: float(v) if isinstance(v, np.floating) else v 
                           for k, v in result['metrics'].items()}
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {results_path}")


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names=None):
    """
    Main function to train and evaluate all models
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        feature_names: List of feature names for importance analysis
    
    Returns:
        trainer: ModelTrainer object with all trained models
    """
    trainer = ModelTrainer()
    
    # Train all models
    trainer.train_all_models(X_train, y_train)
    
    # Evaluate all models
    trainer.evaluate_all_models(X_test, y_test)
    
    # Get and display comparison
    comparison_df = trainer.get_comparison_results()
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Get best model
    best_model_name, best_r2 = trainer.get_best_model()
    print(f"\n🏆 Best Model: {best_model_name.replace('_', ' ').title()} (R² = {best_r2:.4f})")
    
    # Feature importance analysis
    # Feature importance analysis (best-effort - don't block saving)
    if feature_names is not None:
        try:
            print("\n" + "="*60)
            print("FEATURE IMPORTANCE (Top 10 - Random Forest)")
            print("="*60)
            importance_df = trainer.models['random_forest'].get_feature_importance(feature_names)
            print(importance_df.head(10).to_string(index=False))
        except Exception as ex:
            print("Warning: could not compute feature importance:", ex)

    # Save models and results (ensure this always happens)
    try:
        trainer.save_all_models()
        trainer.save_results()
    except Exception as ex:
        print("Warning: error while saving models/results:", ex)
    
    return trainer


if __name__ == '__main__':
    from data_generation import generate_carbon_footprint_dataset
    from data_preprocessing import preprocess_dataset
    
    # Generate data
    print("Generating dataset...")
    df = generate_carbon_footprint_dataset(n_samples=1000)
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(df)
    
    # Train and evaluate models
    print("Training and evaluating models...")
    trainer = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        feature_names=preprocessor.feature_names
    )
