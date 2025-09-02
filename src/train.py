
import pandas as pd
import joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

MODEL_DIR = os.path.join("models")

def train_and_save_models(data_path="data/Telco-Customer-Churn.csv"):
    df = pd.read_csv(data_path)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_mask = df['TotalCharges'].isna()
    df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'MonthlyCharges'] * df.loc[missing_mask, 'tenure']
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    artefacts = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    preprocessor = artefacts['preprocessor']
    X = df.drop(columns=['Churn'])
    y = df['Churn'].values

    X_trans = preprocessor.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression baseline
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    y_pred_proba = log.predict_proba(X_test)[:,1]
    log_auc = roc_auc_score(y_test, y_pred_proba)
    log_acc = accuracy_score(y_test, log.predict(X_test))

    print(f"Logistic Regression AUC: {log_auc:.4f} Acc: {log_acc:.4f}")

    # Random Forest with small grid search
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {'n_estimators':[100,200], 'max_depth':[6,10]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    y_pred_proba_rf = best_rf.predict_proba(X_test)[:,1]
    rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
    rf_acc = accuracy_score(y_test, best_rf.predict(X_test))
    print(f"RandomForest best params: {grid.best_params_}")
    print(f"RandomForest AUC: {rf_auc:.4f} Acc: {rf_acc:.4f}")

    # XGBoost (if available)
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_proba_xgb = xgb.predict_proba(X_test)[:,1]
        xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
        xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
        print(f"XGBoost AUC: {xgb_auc:.4f} Acc: {xgb_acc:.4f}")
    except Exception as e:
        print("XGBoost not available or failed to train:", e)
        xgb = None

    # Save best model by AUC
    aucs = {'log': log_auc, 'rf': rf_auc, 'xgb': xgb_auc if xgb is not None else 0}
    best_name = max(aucs, key=aucs.get)
    best_model = {'log': log, 'rf': best_rf, 'xgb': xgb}[best_name]
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'model': best_model, 'model_name': best_name}, os.path.join(MODEL_DIR, 'best_model.joblib'))
    print(f"Saved best model ({best_name}) to {MODEL_DIR}/best_model.joblib")
