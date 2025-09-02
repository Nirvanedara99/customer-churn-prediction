
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

MODEL_DIR = os.path.join("models")

def load_data(path="data/Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_and_save(path="data/Telco-Customer-Churn.csv"):
    df = load_data(path)
    # drop customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    # strip spaces in column names if any
    df.columns = [c.strip() for c in df.columns]

    # Clean TotalCharges: convert to numeric, coerce errors
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing TotalCharges with MonthlyCharges * tenure as heuristic
    if 'TotalCharges' in df.columns:
        missing_mask = df['TotalCharges'].isna()
        df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'MonthlyCharges'] * df.loc[missing_mask, 'tenure']

    # Target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Build preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # fit preprocessor and save
    preprocessor.fit(df.drop(columns=['Churn']))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'preprocessor': preprocessor, 'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols}, os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    print(f"Saved preprocessor with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical cols to {MODEL_DIR}/preprocessor.joblib")
