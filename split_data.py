import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import pickle

# Load preprocessed data
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load and preprocess the data
df = pd.read_csv('weatherHistory.csv')
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

# Create time-based features
df['Hour'] = df['Formatted Date'].dt.hour
df['Month'] = df['Formatted Date'].dt.month
df['DayOfWeek'] = df['Formatted Date'].dt.dayofweek

# Create target variable (next hour temperature)
df['Next_Temperature'] = df['Temperature (C)'].shift(-1)
df = df.dropna(subset=['Next_Temperature'])

# Prepare features and target
X = df.drop(['Next_Temperature', 'Formatted Date', 'Daily Summary'], axis=1)
y = df['Next_Temperature']

# Transform features
X_transformed = preprocessor.transform(X)

# Create DataFrame with transformed features
X_df = pd.DataFrame(X_transformed, columns=feature_names)

# Sort by date for time series split
dates = df['Formatted Date']
X_df['date'] = dates.values
X_df = X_df.sort_values('date')
y = y[X_df.index]

# Remove date column after sorting
X_df = X_df.drop('date', axis=1)

# Calculate split points
total_samples = len(X_df)
train_size = int(0.7 * total_samples)
val_size = int(0.15 * total_samples)

# Create splits while preserving temporal order
X_train = X_df.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X_df.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]

X_test = X_df.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

print("=== Data Splitting Summary ===")
print(f"\nTotal samples: {total_samples}")
print(f"Training samples: {len(X_train)} ({len(X_train)/total_samples*100:.1f}%)")
print(f"Validation samples: {len(X_val)} ({len(X_val)/total_samples*100:.1f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/total_samples*100:.1f}%)")
print(f"\nFeatures in split data: {X_df.shape[1]}")

print("\n=== Splitting Strategy ===")
print("1. Time-aware splitting: Data sorted chronologically")
print("2. Split ratios: 70% train, 15% validation, 15% test")
print("3. Preserved temporal ordering to prevent data leakage")
print("4. No shuffling to maintain time series integrity")

# Save splits for model training
splits = {
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val,
    'X_test': X_test, 'y_test': y_test
}

with open('data_splits.pkl', 'wb') as f:
    pickle.dump(splits, f)

print("\nSplit data saved to 'data_splits.pkl'")
