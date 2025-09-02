import sys
import pandas as pd
import joblib

def main(input_csv):
    # Load preprocessor and model
    preprocessor_data = joblib.load("models/preprocessor.joblib")
    model_data = joblib.load("models/best_model.joblib")

    # Extract objects if loaded as dict
    if isinstance(preprocessor_data, dict):
        preprocessor = preprocessor_data.get('preprocessor')
    else:
        preprocessor = preprocessor_data

    if isinstance(model_data, dict):
        model = model_data.get('model')
    else:
        model = model_data

    if preprocessor is None or model is None:
        raise ValueError("Preprocessor or model not found in loaded joblib files.")

    # Read new data
    new_data = pd.read_csv(input_csv)

    if new_data.empty:
        raise ValueError(f"The file {input_csv} is empty. Add data before running predictions.")

    # Transform data
    X_new = preprocessor.transform(new_data)

    # Predict
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]

    # Combine results
    results = new_data.copy()
    results['Churn_Prediction'] = predictions
    results['Churn_Probability'] = probabilities

    # Save
    output_file = "predictions.csv"
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_new_customers.csv>")
    else:
        main(sys.argv[1])
