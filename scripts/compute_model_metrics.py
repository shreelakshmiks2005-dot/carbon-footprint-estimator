"""Compute metrics for existing saved models and write `models/model_results.json`.

This script does NOT retrain models. It loads saved model files (joblib for
sklearn models, H5 for Keras ANN if present), computes metrics on a test
split of the available dataset, and saves a JSON with MAE/RMSE/R2/MAPE for
each model found.

Run from project root:
    python scripts\compute_model_metrics.py
"""

import json
from pathlib import Path
import sys
# Ensure project root and src/ are on sys.path so unpickling can find local modules
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_dataset():
    candidates = [Path('data/carbon_footprint_data_enriched.csv'), Path('data/carbon_footprint_data.csv')]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_preprocessor():
    p = Path('models/preprocessor.pkl')
    if p.exists():
        return joblib.load(str(p))
    return None


def load_model_candidates(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                return joblib.load(str(p))
            except Exception:
                # joblib load failed (maybe not a sklearn model)
                try:
                    import tensorflow as tf
                    return tf.keras.models.load_model(str(p), compile=False)
                except Exception:
                    continue
    return None


def try_load_ann(path):
    p = Path(path)
    if p.exists():
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(str(p), compile=False)
        except Exception:
            return None
    return None


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    try:
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    except Exception:
        mape = None
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def main():
    ds = find_dataset()
    if ds is None:
        print('No dataset found in data/. Place dataset in data/ and retry.')
        return

    print(f'Using dataset: {ds}')
    df = pd.read_csv(ds)

    # find target
    target_col = None
    for col in df.columns:
        if 'carbon' in col.lower() or 'footprint' in col.lower():
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = load_preprocessor()
    if preprocessor is not None:
        try:
            X_test_proc = preprocessor.transform(X_test.copy())
        except Exception as e:
            print('Preprocessor transform failed:', e)
            print('Falling back to numeric coercion on X_test')
            X_test_proc = X_test.copy().apply(pd.to_numeric, errors='coerce').fillna(0).values
    else:
        print('No preprocessor found, using raw numeric conversion for X_test')
        X_test_proc = X_test.copy().apply(pd.to_numeric, errors='coerce').fillna(0).values

    results = {}

    # Linear
    linear_candidates = ['models/linear_model.pkl', 'models/linear_regression.pkl']
    linear = load_model_candidates(linear_candidates)
    if linear is not None:
        try:
            y_pred = linear.predict(X_test_proc)
            results['linear_regression'] = {'metrics': compute_metrics(y_test, y_pred)}
            print('Computed metrics for Linear Regression')
        except Exception as ex:
            print('Linear predict failed:', ex)

    # Random Forest
    rf_candidates = ['models/random_forest_model.pkl', 'models/random_forest.pkl']
    rf = load_model_candidates(rf_candidates)
    if rf is not None:
        try:
            y_pred = rf.predict(X_test_proc)
            results['random_forest'] = {'metrics': compute_metrics(y_test, y_pred)}
            print('Computed metrics for Random Forest')
        except Exception as ex:
            print('Random Forest predict failed:', ex)

    # XGBoost
    xgb_candidates = ['models/xgboost_model.pkl', 'models/xgboost.pkl', 'models/xgboost_model.joblib']
    xgb = load_model_candidates(xgb_candidates)
    if xgb is not None:
        try:
            y_pred = xgb.predict(X_test_proc)
            results['xgboost'] = {'metrics': compute_metrics(y_test, y_pred)}
            print('Computed metrics for XGBoost')
        except Exception as ex:
            print('XGBoost predict failed:', ex)

    # ANN (optional) - include but user already has NN
    ann = try_load_ann('models/ann_model.h5')
    if ann is not None:
        try:
            y_pred = ann.predict(X_test_proc, verbose=0).flatten()
            results['neural_network'] = {'metrics': compute_metrics(y_test, y_pred)}
            print('Computed metrics for Neural Network')
        except Exception as ex:
            print('ANN predict failed:', ex)

    if not results:
        print('No models found to evaluate. Place model files in models/ and retry.')
        return

    out_path = Path('models/model_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=4)

    print('Wrote', out_path)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
