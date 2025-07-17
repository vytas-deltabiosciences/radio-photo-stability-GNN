# generate_scaler.py

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Path to your Excel dataset — adjust if needed
DATA_PATH = "data/curated_data.xlsx"
NUM_FEATURES = ['I', 'T', 'D', 'C0', 'pH']

# Load the dataset
df = pd.read_excel(DATA_PATH)

# Ensure numeric types
df = df.dropna(subset=NUM_FEATURES)
df[NUM_FEATURES] = df[NUM_FEATURES].astype(float)

# Fit the scaler
scaler = StandardScaler().fit(df[NUM_FEATURES])

# Save to file
joblib.dump(scaler, "trained_scaler.pkl")
print("✅  trained_scaler.pkl saved.")
