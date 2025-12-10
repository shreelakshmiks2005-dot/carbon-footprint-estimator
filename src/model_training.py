"""
Model Training - Neural Network for Carbon Footprint Estimation
"""

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import json

class CarbonFootprintModel:
    """Neural Network model for carbon footprint prediction"""
    
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """Build ANN architecture"""
        self.model = Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the neural network"""
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                __import__('tensorflow.keras.callbacks', fromlist=['EarlyStopping']).EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                )
            ]
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, y_pred
    
    def save(self, filepath):
        """Save the model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
    
    @staticmethod
    def load(filepath):
        """Load saved model"""
        from tensorflow.keras.models import load_model
        # When loading a model for inference in a different environment or
        # Keras/TensorFlow version, avoid loading the optimizer/metrics state
        # which can cause deserialization errors for certain metric names.
        return load_model(filepath, compile=False)


def train_carbon_model(X_train, y_train, X_test, y_test):
    """Train and evaluate the ANN model"""
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Build and train model
    model = CarbonFootprintModel()
    model.build_model(input_shape=X_train_split.shape[1])
    
    print("Training Neural Network Model...")
    model.train(X_train_split, y_train_split, X_val, y_val, epochs=100, batch_size=32)
    
    # Evaluate
    metrics, predictions = model.evaluate(X_test, y_test)
    
    print("\n=== ANN Model Performance ===")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2 Score: {metrics['R2']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    
    # Save model
    model_path = Path('models/ann_model.h5')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    
    # Save metrics
    results = {
        'Neural_Network': {
            'metrics': metrics
        }
    }
    
    with open('models/model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nModel saved to {model_path}")
    
    return model, metrics


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

    # Train and evaluate ANN model separately
    print("Training and evaluating ANN model...")
    ann_model, ann_metrics = train_carbon_model(X_train, y_train, X_test, y_test)
