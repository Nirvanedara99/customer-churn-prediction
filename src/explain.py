
import joblib, os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = os.path.join("models")

def explain_saved_model(data_path="data/Telco-Customer-Churn.csv"):
    artefacts = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    preprocessor = artefacts['preprocessor']
    num_cols = artefacts['numeric_cols']
    cat_cols = artefacts['categorical_cols']
    feature_names_num = num_cols
    # For OneHotEncoder feature names, try to build names (best-effort)
    try:
        cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        ohe_names = cat_ohe.get_feature_names_out(cat_cols).tolist()
    except Exception:
        ohe_names = cat_cols
    feature_names = feature_names_num + ohe_names

    df = pd.read_csv(data_path)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_mask = df['TotalCharges'].isna()
    df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'MonthlyCharges'] * df.loc[missing_mask, 'tenure']
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    X = df.drop(columns=['Churn'])
    X_trans = preprocessor.transform(X)

    saved = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
    model = saved['model']

    # Use a small sample to compute SHAP (faster)
    sample = X_trans[np.random.choice(X_trans.shape[0], size=min(500, X_trans.shape[0]), replace=False)]
    try:
        explainer = shap.Explainer(model.predict_proba, sample)
        shap_values = explainer(sample)
        # summary plot
        plt.figure()
        shap.summary_plot(shap_values, features=sample, feature_names=feature_names, show=False)
        plt.savefig(os.path.join(MODEL_DIR, "shap_summary.png"), bbox_inches='tight')
        print(f"Saved SHAP summary to {MODEL_DIR}/shap_summary.png")
    except Exception as e:
        print('SHAP explanation failed:', e)
