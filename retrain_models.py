"""
Retrain models with ANN
"""

import sys
from pathlib import Path

# Fix the path - remove '/src' since we're in root
sys.path.append(str(Path(__file__).parent / 'src'))

from data_generation import generate_carbon_footprint_dataset
from data_preprocessing import DataPreprocessor
from model_training import train_carbon_model
from sklearn.model_selection import train_test_split

# Generate dataset
print("Generating dataset...")
df = generate_carbon_footprint_dataset(n_samples=1000)

# Print column names to check
print("Dataset columns:", df.columns.tolist())
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Find the target column name
target_col = None
for col in df.columns:
    if 'carbon' in col.lower() or 'footprint' in col.lower():
        target_col = col
        break

if target_col is None:
    # If no carbon/footprint column found, use the last column
    target_col = df.columns[-1]

print(f"\nUsing target column: {target_col}")

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess
print("Preprocessing data...")
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor
preprocessor.save('models/preprocessor.pkl')

# Train ANN model
print("Training ANN model...")
model, metrics = train_carbon_model(X_train_processed, y_train, X_test_processed, y_test)

print("\n✅ Training complete! Models saved.")