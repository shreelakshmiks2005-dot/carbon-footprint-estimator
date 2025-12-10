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


class ModelTrainer:
    """Train and evaluate a set of baseline models for comparison"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def train_all_models(self, X_train, y_train):
        """Train Linear Regression, Random Forest, and XGBoost (if available)."""
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['linear_regression'] = lr

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        # Gradient Boosting (sklearn) as a lightweight alternative
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
            gbr.fit(X_train, y_train)
            self.models['gradient_boosting'] = gbr
        except Exception:
            # If sklearn's GradientBoosting is not available (rare), skip
            pass

        # XGBoost (optional) - train if xgboost is installed
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
            xgb.fit(X_train, y_train)
            self.models['xgboost'] = xgb
        except Exception:
            # XGBoost not available — skip
            pass

    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models and store metrics in self.results."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np

        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
            except Exception:
                # Some models (like XGBoost) may return dmatrix outputs; coerce
                y_pred = model.predict(X_test)

            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            # Avoid division by zero in MAPE
            try:
                mape = float((np.mean(np.abs((y_test - y_pred) / y_test))) * 100)
            except Exception:
                mape = None

            self.results[name] = {
                'metrics': {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAPE': mape
                }
            }

    def get_comparison_results(self):
        """Return a pandas DataFrame suitable for display in the app."""
        import pandas as pd

        rows = []
        for name, val in self.results.items():
            m = val.get('metrics', {})
            rows.append({
                'Model': name.replace('_', ' ').title(),
                'MAE': m.get('MAE'),
                'RMSE': m.get('RMSE'),
                'R2': m.get('R2'),
                'MAPE (%)': m.get('MAPE')
            })

        if rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'R2', 'MAPE (%)'])

    def get_best_model(self):
        """Return the name and R2 of the best model by R2 score."""
        best_name = None
        best_r2 = -float('inf')
        for name, val in self.results.items():
            r2 = val.get('metrics', {}).get('R2')
            if r2 is not None and r2 > best_r2:
                best_r2 = r2
                best_name = name
        return best_name, best_r2

    def save_all_models(self):
        """Save non-Keras models to `models/` using joblib."""
        import joblib
        Path('models').mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            try:
                joblib.dump(model, f'models/{name}.pkl')
            except Exception:
                # If a model fails to serialize with joblib, skip
                continue

    def save_results(self, filepath='models/model_results.json'):
        import json
        Path('models').mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as fh:
            json.dump(self.results, fh, indent=4)

    def get_feature_importance(self, feature_names):
        """Return feature importance DataFrame for Random Forest if available."""
        import pandas as pd
        rf = self.models.get('random_forest')
        if rf is None:
            raise ValueError('Random Forest model not trained')
        try:
            import numpy as np
            imp = rf.feature_importances_
            df = pd.DataFrame({'feature': feature_names, 'importance': imp})
            return df.sort_values('importance', ascending=False)
        except Exception as ex:
            raise ex


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
