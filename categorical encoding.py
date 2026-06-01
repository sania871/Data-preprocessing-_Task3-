# -------------------------------
# Dataset: crop_yield.csv
# -------------------------------

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from scipy.stats import skew

# 1. Load and understand dataset
path = r"C:\Users\Sania\Downloads\archive\crop_yield.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("Columns:", df.columns)
print("First 5 rows:\n", df.head())

# 2. Handle missing values
print("\nMissing values per column:\n", df.isnull().sum())
df = df.dropna(subset=['yield'])   # Example: drop rows where target 'yield' is missing
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric NaNs with median
df.fillna("Unknown", inplace=True)  # Fill categorical NaNs with 'Unknown'

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Detect and handle outliers (IQR method for numeric features)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df[col] = np.where(df[col] < lower, lower,
              np.where(df[col] > upper, upper, df[col]))

# 5. Handle incorrect data types
df['year'] = pd.to_numeric(df['year'], errors='coerce')   # Convert year to numeric
df['crop'] = df['crop'].astype(str)                       # Ensure crop is string

# 6. Handle categorical variables
# Example: crop type (nominal) → One-Hot Encoding
df = pd.get_dummies(df, columns=['crop'], drop_first=True)

# Example: season (ordinal: Kharif < Rabi < Zaid)
if 'season' in df.columns:
    df['season'] = df['season'].str.strip() # Clean whitespace from season column
    season_order = [['Kharif', 'Rabi', 'Zaid']]
    # Handle unknown categories by assigning them a value (e.g., -1)
    oe = OrdinalEncoder(categories=season_order, handle_unknown='use_encoded_value', unknown_value=-1)
    df['season_encoded'] = oe.fit_transform(df[['season']])
    df.drop('season', axis=1, inplace=True)

# Example: irrigation (binary Yes/No) → Label Encoding
if 'irrigation' in df.columns:
    le = LabelEncoder()
    df['irrigation_encoded'] = le.fit_transform(df['irrigation'])
    df.drop('irrigation', axis=1, inplace=True)

# 7. Feature scaling
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns # Re-evaluate numeric_cols after one-hot encoding
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 8. Remove irrelevant/redundant features
irrelevant = ['farmer_name', 'record_id']  # Example columns
df = df.drop(columns=[col for col in irrelevant if col in df.columns])

# 9. Handle skewness
for col in numeric_cols:
    if abs(skew(df[col])) > 1:   # Threshold for skewness
        df[col] = np.log1p(df[col] - df[col].min() + 1)  # log transform

print("\n Final cleaned dataset ready for modeling")
print("Shape after preprocessing:", df.shape)
