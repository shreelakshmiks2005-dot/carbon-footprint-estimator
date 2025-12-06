"""
Helper to import a CSV (local or downloaded from Kaggle) and map it to the
project's expected dataset schema: `data/carbon_footprint_data.csv`.

Usage examples:

# 1) Import local CSV and produce mapping suggestions
python -m src.kaggle_import --input path/to/kaggle_dataset.csv

# 2) Use a mapping file (after editing suggestions) to produce standardized CSV
python -m src.kaggle_import --input path/to/kaggle_dataset.csv --mapping mapping.json --output data/carbon_footprint_data.csv

# 3) Download from Kaggle (optional, requires `kaggle` package and Kaggle API token)
python -m src.kaggle_import --kaggle-dataset zillow/zecon --file dataset.csv --output data/carbon_footprint_data.csv

This script attempts to auto-detect columns by keywords and writes a
`kaggle_mapping_suggestions.json` file with suggested mappings you can edit.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import re

EXPECTED_COLUMNS = [
    'transportation_mode',
    'commute_distance_km',
    'flights_per_year',
    'diet_type',
    'meat_consumption_kg_per_month',
    'electricity_usage_kwh',
    'gas_usage_therms',
    'water_usage_gallons',
    'recycling_percentage',
    'new_clothes_per_month',
    'car_type',
    'monthly_carbon_footprint_kg'  # optional if you have labels
]

KEYWORD_MAP = {
    'transportation_mode': ['transport', 'transportation_mode', 'mode'],
    'commute_distance_km': ['distance', 'commute', 'km', 'miles', 'mile'],
    'flights_per_year': ['flight', 'flights', 'air_travel', 'airplane'],
    'diet_type': ['diet', 'diet_type', 'food_type'],
    'meat_consumption_kg_per_month': ['meat', 'meat_consumption', 'meat_kg'],
    'electricity_usage_kwh': ['electricity', 'kwh', 'electric_usage', 'electric_use'],
    'gas_usage_therms': ['gas', 'therm', 'natural_gas'],
    'water_usage_gallons': ['water', 'gallon', 'litre', 'litre', 'liters', 'gallons'],
    'recycling_percentage': ['recycle', 'recycling', 'recycled', 'recycling_percentage'],
    'new_clothes_per_month': ['clothes', 'clothing', 'new_clothes', 'apparel'],
    'car_type': ['car', 'vehicle', 'car_type', 'vehicle_type'],
    'monthly_carbon_footprint_kg': ['carbon', 'footprint', 'co2', 'co2_kg', 'monthly_carbon']
}


def normalize_col(col):
    col = col.lower()
    col = re.sub(r'[^a-z0-9_]', '_', col)
    col = re.sub(r'__+', '_', col)
    return col


def suggest_mapping(df_cols):
    suggestions = {}
    norm_cols = {c: normalize_col(c) for c in df_cols}

    for expected in EXPECTED_COLUMNS:
        candidates = []
        keywords = KEYWORD_MAP.get(expected, [expected])
        for orig, norm in norm_cols.items():
            for kw in keywords:
                if kw in norm:
                    candidates.append(orig)
                    break
        # also fuzzy fallback: look for numeric columns if expected is numeric
        if not candidates:
            if expected in ['commute_distance_km', 'flights_per_year',
                            'meat_consumption_kg_per_month', 'electricity_usage_kwh',
                            'gas_usage_therms', 'water_usage_gallons',
                            'recycling_percentage', 'new_clothes_per_month',
                            'monthly_carbon_footprint_kg']:
                # choose first numeric column not already mapped
                candidates = []
        suggestions[expected] = candidates[0] if candidates else None
    return suggestions


def apply_mapping(df, mapping):
    # Create new df with expected columns (if present in mapping)
    out = pd.DataFrame()
    for expected in EXPECTED_COLUMNS:
        src = mapping.get(expected)
        if src and src in df.columns:
            out[expected] = df[src]
        else:
            # fill missing numeric columns with zeros or sensible defaults
            if expected == 'transportation_mode':
                out[expected] = 'Car'
            elif expected == 'diet_type':
                out[expected] = 'Meat-based'
            elif expected == 'car_type':
                out[expected] = 'Petrol'
            else:
                out[expected] = 0
    return out


def download_kaggle_dataset(slug, file_name, dest_dir=Path('kaggle_downloads')):
    """Attempt to download a dataset file using kaggle API.
    Requires `kaggle` package and ~/.kaggle/kaggle.json
    Returns path to downloaded file or None if failed."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("Kaggle API not available. Install with: pip install kaggle and set up API token.")
        return None

    api = KaggleApi()
    api.authenticate()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {slug} -> {file_name} ...")
        api.dataset_download_file(slug, file_name, path=str(dest_dir), force=True)
        zip_path = dest_dir / (file_name + '.zip')
        if zip_path.exists():
            # if kaggle returns a zip, try to extract
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(dest_dir)
            csv_path = dest_dir / file_name
            if csv_path.exists():
                return str(csv_path)
        else:
            csv_path = dest_dir / file_name
            if csv_path.exists():
                return str(csv_path)
    except Exception as e:
        print(f"Kaggle download failed: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Path to local CSV file')
    parser.add_argument('--kaggle-dataset', '-k', help='Kaggle dataset slug (owner/dataset)')
    parser.add_argument('--file', '-f', help='Filename inside Kaggle dataset to download')
    parser.add_argument('--mapping', '-m', help='Path to mapping JSON file (optional)')
    parser.add_argument('--output', '-o', default='data/carbon_footprint_data.csv', help='Output path')

    args = parser.parse_args()

    input_path = None
    if args.kaggle_dataset and args.file:
        downloaded = download_kaggle_dataset(args.kaggle_dataset, args.file)
        if downloaded:
            input_path = downloaded
        else:
            print("Failed to download from Kaggle. Provide a local --input CSV instead.")
            return
    elif args.input:
        input_path = args.input
    else:
        print("Provide either --input path/to.csv or --kaggle-dataset owner/dataset --file filename.csv")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

    suggested = suggest_mapping(df.columns.tolist())
    suggestion_path = Path('kaggle_mapping_suggestions.json')
    with suggestion_path.open('w', encoding='utf8') as f:
        json.dump(suggested, f, indent=2)

    print(f"Wrote column mapping suggestions to {suggestion_path}")
    print("Suggested mapping (expected_column: detected_column)")
    for k, v in suggested.items():
        print(f"  {k}: {v}")

    if args.mapping:
        mapping_file = Path(args.mapping)
        if mapping_file.exists():
            mapping = json.load(mapping_file.open('r', encoding='utf8'))
        else:
            print(f"Mapping file {mapping_file} not found. Using suggestions.")
            mapping = suggested
    else:
        # by default use suggestions
        mapping = suggested

    standardized = apply_mapping(df, mapping)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(out_path, index=False)
    print(f"Standardized dataset saved to {out_path}")
    print("Preview of output columns:")
    print(standardized.head())


if __name__ == '__main__':
    main()
