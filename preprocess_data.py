import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the data
df = pd.read_csv('weatherHistory.csv')

# Convert datetime with UTC=True
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)

# Create time-based features
df['Hour'] = df['Formatted Date'].dt.hour
df['Month'] = df['Formatted Date'].dt.month
df['DayOfWeek'] = df['Formatted Date'].dt.dayofweek

# Create target variable (next hour temperature)
df['Next_Temperature'] = df['Temperature (C)'].shift(-1)

# Remove the last row since it won't have a next hour temperature
df = df.dropna(subset=['Next_Temperature'])

# Calculate missing value statistics before preprocessing
total_rows = len(df)
missing_stats = df.isnull().sum()
missing_percentage = (missing_stats / total_rows) * 100

print("=== Missing Values Analysis ===")
print("\nTotal data points:", total_rows)
print("\nMissing values by feature:")
for column in df.columns:
    missing_count = missing_stats[column]
    if missing_count > 0:
        print(f"{column}: {missing_count} values ({missing_percentage[column]:.2f}%)")

# Identify numeric and categorical columns
numeric_features = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
                   'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
                   'Pressure (millibars)', 'Hour', 'Month', 'DayOfWeek']

categorical_features = ['Summary', 'Precip Type']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Initialize and fit the preprocessor
print("\n=== Preprocessing Data ===")
print("Features before preprocessing:", df.shape[1])
print("Data points before preprocessing:", df.shape[0])

# Fit preprocessor and transform data
X = df.drop(['Next_Temperature', 'Formatted Date', 'Daily Summary'], axis=1)
y = df['Next_Temperature']

X_transformed = preprocessor.fit_transform(X)

# Get feature names after preprocessing
onehot_features = []
for feature, categories in zip(categorical_features,
                             preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_):
    onehot_features.extend([f"{feature}_{cat}" for cat in categories])

feature_names = numeric_features + onehot_features

print("\nFeatures after preprocessing:", X_transformed.shape[1])
print("Data points after preprocessing:", X_transformed.shape[0])

# Save preprocessor and feature names
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\n=== Preprocessing Details ===")
print("\nPreprocessing steps applied:")
print("1. Numeric features: StandardScaler")
print("2. Categorical features: OneHotEncoder")
print("3. Time-based features created: Hour, Month, DayOfWeek")
print("4. Target variable: Next hour temperature (shifted Temperature (C))")
print("\nFeatures used:")
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
