import traceback
import pandas as pd
from pathlib import Path
import sys

# Ensure project root is on sys.path so `import src...` works for unpickling
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preprocessing import DataPreprocessor
from src.model_training import CarbonFootprintModel


def run_test():
    try:
        preprocessor = DataPreprocessor.load('models/preprocessor.pkl')
        model = CarbonFootprintModel.load('models/xgboost_model.pkl')
        print('Loaded preprocessor and model successfully')

        sample = {
            'transportation_mode': 'private',
            'commute_distance_km': 20,
            'flights_per_year': 1,
            'diet_type': 'omnivore',
            'meat_consumption_kg_per_month': 8.0,
            'electricity_usage_kwh': 350,
            'gas_usage_therms': 30,
            'water_usage_gallons': 2000,
            'recycling_percentage': 50,
            'new_clothes_per_month': 2,
            'car_type': 'electric',
            'household_size': 2,
            'vehicle_fuel_efficiency_km_per_l': 0.0,
            'renewable_energy_percentage': 20,
            'heating_fuel_type': 'electric'
        }

        df = pd.DataFrame([sample])

        processed = preprocessor.transform(df)
        print('Preprocessor.transform succeeded. Processed shape:', processed.shape)

        pred = model.predict(processed)
        print('Model prediction:', pred)

    except Exception as e:
        print('ERROR during test prediction:')
        traceback.print_exc()


if __name__ == '__main__':
    run_test()
