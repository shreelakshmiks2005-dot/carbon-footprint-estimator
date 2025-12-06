"""
Configuration and Usage Guide for Carbon Footprint Estimator
"""

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Dataset Configuration
DATASET_CONFIG = {
    'n_samples': 1000,              # Number of synthetic records to generate
    'random_state': 42,             # Random seed for reproducibility
    'test_size': 0.2,               # Test-train split ratio
    'output_path': 'data/carbon_footprint_data.csv'
}

# Model Configuration
MODEL_CONFIG = {
    'linear_regression': {
        'name': 'Linear Regression',
        'params': {}
    },
    'random_forest': {
        'name': 'Random Forest',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'xgboost': {
        'name': 'XGBoost',
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    }
}

# Feature Configuration
FEATURES = {
    'categorical': [
        'transportation_mode',
        'diet_type',
        'car_type'
    ],
    'numerical': [
        'commute_distance_km',
        'flights_per_year',
        'meat_consumption_kg_per_month',
        'electricity_usage_kwh',
        'gas_usage_therms',
        'water_usage_gallons',
        'recycling_percentage',
        'new_clothes_per_month'
    ],
    'engineered': [
        'high_transport_user',
        'high_energy_user',
        'sustainability_score',
        'waste_water_factor'
    ]
}

# Carbon Emission Factors (kg CO2 equivalents)
EMISSION_FACTORS = {
    'transportation': {
        'car_petrol': 0.21,              # kg CO2 per km
        'car_diesel': 0.18,
        'car_hybrid': 0.10,
        'car_electric': 0.05,
        'public_transport': 0.05,
        'flight': 0.255,                 # kg CO2 per km (average per person)
        'bike_walk': 0.0
    },
    'energy': {
        'electricity': 0.92,             # kg CO2 per kWh
        'natural_gas': 5.3               # kg CO2 per therm
    },
    'diet': {
        'meat': 27,                      # kg CO2 per kg of meat
        'vegetarian': 12,                # kg CO2 per kg equivalent
        'vegan': 2                       # kg CO2 per kg equivalent
    },
    'other': {
        'water': 0.35,                   # kg CO2 per gallon
        'clothing': 6.5,                 # kg CO2 per new item
        'recycling_offset': 0.1          # kg CO2 offset per 1% recycled
    }
}

# Sustainability Level Thresholds
SUSTAINABILITY_LEVELS = {
    'transportation': {
        'high': {'score': 5, 'description': 'Above 20km daily commute or >2 flights/year'},
        'medium': {'score': 2, 'description': 'Moderate transport usage'},
        'low': {'score': 0, 'description': 'Below average transport emissions'}
    },
    'energy': {
        'high': {'threshold': 550, 'description': 'Above 550 kWh equivalent per month'},
        'medium': {'threshold': 350, 'description': 'Moderate energy usage'},
        'low': {'threshold': 0, 'description': 'Below 350 kWh equivalent per month'}
    },
    'diet': {
        'high': {'description': 'Meat-based diet with >10 kg/month consumption'},
        'medium': {'description': 'Meat-based with moderate consumption or vegetarian'},
        'low': {'description': 'Vegan or plant-based diet'}
    }
}

# Recommendation Categories
RECOMMENDATION_CATEGORIES = {
    'transportation': {
        'emoji': '',
        'description': 'Commuting and travel habits',
        'high_impact_changes': [
            'Switch to public transportation',
            'Consider electric vehicle',
            'Carpool with colleagues',
            'Reduce number of flights'
        ]
    },
    'diet': {
        'emoji': '',
        'description': 'Food and diet choices',
        'high_impact_changes': [
            'Introduce Meatless Mondays',
            'Reduce meat consumption',
            'Buy local and seasonal',
            'Consider plant-based alternatives'
        ]
    },
    'energy': {
        'emoji': '',
        'description': 'Electricity and gas usage',
        'high_impact_changes': [
            'Switch to renewable energy',
            'Install solar panels',
            'Use LED bulbs',
            'Install smart thermostat'
        ]
    },
    'waste': {
        'emoji': '',
        'description': 'Waste management and consumption',
        'high_impact_changes': [
            'Increase recycling rate',
            'Compost organic waste',
            'Buy secondhand items',
            'Reduce new clothing purchases'
        ]
    },
    'water': {
        'emoji': '',
        'description': 'Water consumption and conservation',
        'high_impact_changes': [
            'Install low-flow showerheads',
            'Fix leaky pipes',
            'Take shorter showers',
            'Use full loads for laundry'
        ]
    }
}

# File Paths
FILE_PATHS = {
    'data': 'data/carbon_footprint_data.csv',
    'models': {
        'linear': 'models/linear_model.pkl',
        'random_forest': 'models/random_forest_model.pkl',
        'xgboost': 'models/xgboost_model.pkl',
        'preprocessor': 'models/preprocessor.pkl',
        'results': 'models/model_results.json'
    },
    'notebook': 'Carbon_Footprint_Analysis.ipynb',
    'readme': 'README.md'
}

# Evaluation Metrics
EVALUATION_METRICS = {
    'MAE': 'Mean Absolute Error - Average absolute difference',
    'RMSE': 'Root Mean Squared Error - Standard deviation of residuals',
    'R2': 'R² Score - Proportion of variance explained (0-1)',
    'MAPE': 'Mean Absolute Percentage Error - Percentage accuracy'
}

# Expected Performance Benchmarks
PERFORMANCE_BENCHMARKS = {
    'linear_regression': {
        'mae': 8.5,
        'rmse': 11.2,
        'r2': 0.75
    },
    'random_forest': {
        'mae': 4.2,
        'rmse': 5.5,
        'r2': 0.92
    },
    'xgboost': {
        'mae': 3.8,
        'rmse': 5.0,
        'r2': 0.94
    }
}

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

USAGE_INSTRUCTIONS = """
╔════════════════════════════════════════════════════════════════════════════╗
║         CARBON FOOTPRINT ESTIMATOR - USAGE INSTRUCTIONS                    ║
╚════════════════════════════════════════════════════════════════════════════╝

1. INSTALLATION
   ───────────────────────────────────────────────────────────────────────
   pip install -r requirements.txt

2. GENERATE DATASET & TRAIN MODELS
   ───────────────────────────────────────────────────────────────────────
   Option A: Streamlit App (Recommended)
   $ streamlit run app.py
   → Navigate to "Model Training" tab
   → Click "Train All Models" button

   Option B: Jupyter Notebook
   $ jupyter notebook Carbon_Footprint_Analysis.ipynb
   → Run all cells sequentially

   Option C: Python Scripts
   $ python src/data_generation.py
   $ python src/model_training.py

3. MAKE PREDICTIONS
   ───────────────────────────────────────────────────────────────────────
   $ streamlit run app.py
   → Go to "Predictions" tab
   → Enter your lifestyle data
   → Click "Get Prediction & Recommendations"
   → View carbon footprint and personalized tips

4. EXPLORE RESULTS
   ───────────────────────────────────────────────────────────────────────
   Available Sections:
   • Dashboard: Statistics and visualizations
   • Model Training: Model comparison and performance
   • Predictions: Personal carbon footprint calculator
   • Analysis: Detailed correlation and trend analysis
   • ℹ️ About: Project information and technologies

5. INTERPRET RESULTS
   ───────────────────────────────────────────────────────────────────────
   • Prediction: Your estimated monthly CO2 emissions in kg
   • Status: Above/Average/Below compared to average user
   • Recommendations: Personalized tips to reduce emissions
   • Potential Savings: Estimated CO2 reduction per category

═══════════════════════════════════════════════════════════════════════════════

INPUT DATA RANGES (for predictions):
  • Commute Distance: 0-50 km
  • Flights per Year: 0-20
  • Electricity Usage: 100-1000 kWh/month
  • Gas Usage: 0-200 therms/month
  • Water Usage: 500-5000 gallons/month
  • Recycling: 0-100%
  • Meat Consumption: 0-20 kg/month
  • New Clothes: 0-10 items/month

═══════════════════════════════════════════════════════════════════════════════
"""

# Data Validation Rules
DATA_VALIDATION = {
    'commute_distance_km': {'min': 0, 'max': 100, 'type': 'float'},
    'flights_per_year': {'min': 0, 'max': 50, 'type': 'int'},
    'meat_consumption_kg_per_month': {'min': 0, 'max': 30, 'type': 'float'},
    'electricity_usage_kwh': {'min': 50, 'max': 2000, 'type': 'float'},
    'gas_usage_therms': {'min': 0, 'max': 500, 'type': 'float'},
    'water_usage_gallons': {'min': 100, 'max': 10000, 'type': 'float'},
    'recycling_percentage': {'min': 0, 'max': 100, 'type': 'float'},
    'new_clothes_per_month': {'min': 0, 'max': 20, 'type': 'float'},
    'transportation_mode': {'values': ['Car', 'Public Transport', 'Bike', 'Walk']},
    'diet_type': {'values': ['Vegan', 'Vegetarian', 'Meat-based']},
    'car_type': {'values': ['Electric', 'Hybrid', 'Petrol', 'Diesel', 'None']}
}

# ============================================================================
# TROUBLESHOOTING GUIDE
# ============================================================================

TROUBLESHOOTING = {
    'models_not_found': {
        'issue': 'ModuleNotFoundError or models not found',
        'solution': [
            '1. Ensure you are in the correct directory: d:\\shree\\projects\\carbon_footprint-main\\cf',
            '2. Run "Model Training" in the Streamlit app first',
            '3. Check if models/ folder exists',
            '4. Try: python src/model_training.py'
        ]
    },
    'streamlit_error': {
        'issue': 'Streamlit app not starting',
        'solution': [
            '1. Verify Streamlit is installed: pip install streamlit',
            '2. Update Streamlit: pip install --upgrade streamlit',
            '3. Check Python version: python --version (need 3.8+)',
            '4. Try: streamlit run app.py --logger.level=debug'
        ]
    },
    'import_errors': {
        'issue': 'ImportError or ModuleNotFoundError',
        'solution': [
            '1. Reinstall dependencies: pip install -r requirements.txt',
            '2. Check if virtual environment is activated',
            '3. Verify Python path: python -m pip list',
            '4. Try: pip install --upgrade scikit-learn xgboost'
        ]
    },
    'data_not_found': {
        'issue': 'Dataset file not found',
        'solution': [
            '1. Check data/ folder exists',
            '2. Generate new data: python src/data_generation.py',
            '3. Use Streamlit app to generate data automatically',
            '4. Verify file path in code'
        ]
    }
}

# Quick command reference
QUICK_COMMANDS = {
    'install': 'pip install -r requirements.txt',
    'app': 'streamlit run app.py',
    'notebook': 'jupyter notebook Carbon_Footprint_Analysis.ipynb',
    'generate_data': 'python src/data_generation.py',
    'train_models': 'python src/model_training.py',
    'quickstart': 'python QUICKSTART.py',
    'list_files': 'ls -la'  # or 'dir' on Windows
}

if __name__ == '__main__':
    print(USAGE_INSTRUCTIONS)
    print("\nFor more information, see README.md")
