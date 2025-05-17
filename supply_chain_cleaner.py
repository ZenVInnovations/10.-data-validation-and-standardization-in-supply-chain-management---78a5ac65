import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def validate_and_standardize_data(input_csv, output_csv, remove_outliers=True):
    # Load data
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"❌ File not found: {input_csv}")
        return

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = imputer.fit_transform(df[[col]])
        elif df[col].dtype in ['float64', 'int64']:
            df[col].fillna(0, inplace=True)

    # Standardize string/text columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Standardize date columns
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Anomaly detection on numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        features = df[numeric_cols]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        iso_model = IsolationForest(contamination=0.1, random_state=42)
        df['Anomaly_Flag'] = iso_model.fit_predict(features_scaled)
        df['Anomaly_Flag'] = df['Anomaly_Flag'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

        if remove_outliers:
            df = df[df['Anomaly_Flag'] == 'Normal'].reset_index(drop=True)

    # Save cleaned and standardized data
    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned data saved to: {output_csv}")
    return df

# Ask user for input file path
input_file = input("Enter the path to your CSV file: ").strip()

# Optional: check if file exists
if not os.path.exists(input_file):
    print("❌ The file does not exist. Please check the path and try again.")
else:
    # Dynamically name output file
    base_filename = os.path.basename(input_file)
    filename_wo_ext = os.path.splitext(base_filename)[0]
    output_file = f"Validated_and_Standardized_{filename_wo_ext}.csv"

    df_cleaned = validate_and_standardize_data(input_file, output_file, remove_outliers=True)

    if df_cleaned is not None:
        print("\n🔍 Preview of cleaned data:")
        print(df_cleaned.head())
