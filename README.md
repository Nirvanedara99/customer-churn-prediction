
# Customer Churn Prediction

**Project structure**:
```
customer-churn-prediction/
├── data/               # Put dataset here: Telco-Customer-Churn.csv
├── models/             # saved model & preprocessors
├── notebooks/          # optional EDA notebooks
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   └── __init__.py
├── requirements.txt
└── pipeline.py
```

## Quick start
1. Place the CSV dataset in `data/` named exactly:
   `data/Telco-Customer-Churn.csv`

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate    # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

3. Run the full pipeline (preprocess → train → evaluate → explain):
```bash
python pipeline.py
```

4. To predict for a single customer, use:
```bash
python src/predict.py --json '{"tenure":12,"MonthlyCharges":70,"Contract":"Month-to-month", "gender":"Female", "SeniorCitizen":0, "Partner":"No", "Dependents":"No", "PhoneService":"Yes", "MultipleLines":"No phone service", "InternetService":"DSL", "OnlineSecurity":"No", "OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","TotalCharges":840}'
```

## What is included
- End-to-end pipeline with:
  - preprocessing and feature engineering
  - multiple-model training and selection
  - model evaluation (metrics + ROC)
  - SHAP-based explainability (summary plot saved)
  - CLI prediction script
- `models/` will contain `preprocessor.joblib` and `best_model.joblib` after running.

## Notes
- This project expects the Telco dataset (standard public dataset). Filename must match.
- If you want deployment (Flask/Streamlit), tell me and I will add a minimal app.
