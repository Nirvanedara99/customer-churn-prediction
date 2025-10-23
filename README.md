**Customer Churn Prediction**
**Project Description**

This project predicts whether a customer is likely to churn using historical customer data. Businesses can use it to identify high-risk customers and take proactive measures to improve retention.

**Features**

Predicts customer churn using machine learning algorithms

Data preprocessing and feature engineering included

Generates performance metrics: accuracy, precision, recall, F1-score

Includes visualizations for insights

**Dataset**

Original dataset: data/Telco-Customer-Churn.csv

High-risk customers: high_risk_customers.csv

New customers for prediction: new_customers.csv

Predicted churn output: predictions.csv

Note: Place your datasets in the customer-churn-prediction/ folder or update paths in scripts accordingly.

**Installation & Requirements**


cd customer-churn-prediction

Create a virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


**Install required packages:**

pip install -r requirements.txt


Example requirements.txt:

pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
joblib

**Usage / How to Run Data Preprocessing:**

python data_preprocessing.py --input data/Telco-Customer-Churn.csv --output processed_data.csv


Cleans and preprocesses the dataset. Outputs preprocessed data as processed_data.csv.

**Train Machine Learning Models:**

python train_model.py --input processed_data.csv --model_output best_model.pkl


Trains multiple models (Logistic Regression, Random Forest, XGBoost)

Saves the best model as best_model.pkl.

**Evaluate Model:**

python evaluate_model.py --model best_model.pkl --test_data processed_data.csv


Loads the saved model and evaluates on test data.

Generates accuracy, precision, recall, and F1-score metrics.

**Predict New Customer Churn:**

python predict_churn.py --model best_model.pkl --input new_customers.csv --output predictions.csv


Predicts churn for new customers and saves the results as predictions.csv.

**Visualizations:**

python visualize_data.py --input processed_data.csv


Generates plots like churn distribution, correlation heatmap, and feature importance.

**Model Details**

Implemented Models:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Best model is automatically selected based on performance metrics.

**Results / Performance Metrics**

**Example results on test data:**

Accuracy: 87%
Precision: 82%
Recall: 79%
F1-score: 80%
