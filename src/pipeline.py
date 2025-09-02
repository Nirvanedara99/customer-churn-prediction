
from src.preprocessing import load_data, preprocess_and_save
from src.train import train_and_save_models
from src.evaluate import evaluate_saved_model
from src.explain import explain_saved_model

DATA_PATH = "data/Telco-Customer-Churn.csv"

def main():
    print("1) Preprocessing...")
    preprocess_and_save(DATA_PATH)
    print("2) Training...")
    train_and_save_models()
    print("3) Evaluation...")
    evaluate_saved_model()
    print("4) Explainability (SHAP)...")
    explain_saved_model()
    print("Pipeline finished. Models and artifacts are in the models/ folder.")

if __name__ == '__main__':
    main()
