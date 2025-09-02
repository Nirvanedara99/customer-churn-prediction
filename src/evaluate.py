
import joblib, os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join("models")

def evaluate_saved_model(data_path="data/Telco-Customer-Churn.csv"):
    artefacts = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    preprocessor = artefacts['preprocessor']
    num_cols = artefacts['numeric_cols']
    cat_cols = artefacts['categorical_cols']

    df = pd.read_csv(data_path)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_mask = df['TotalCharges'].isna()
    df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'MonthlyCharges'] * df.loc[missing_mask, 'tenure']
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    X = df.drop(columns=['Churn'])
    y = df['Churn'].values
    X_trans = preprocessor.transform(X)

    saved = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
    model = saved['model']
    y_pred = model.predict(X_trans)
    y_proba = model.predict_proba(X_trans)[:,1]

    auc = roc_auc_score(y, y_proba)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    print("Evaluation on full dataset:")
    print(f"AUC: {auc:.4f}  Acc: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification report:")
    print(report)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
    print(f"Saved ROC curve to {MODEL_DIR}/roc_curve.png")
