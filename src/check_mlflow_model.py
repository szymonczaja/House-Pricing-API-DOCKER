import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error
import numpy as np
import mlflow
import mlflow.pyfunc
import os
import warnings
import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore", 
    message="This Pipeline instance is not fitted yet", 
    category=FutureWarning
)

TARGET_COLUMN = 'SalePrice' 
DATA_PATH = 'src/data/train.csv' 
EXPERIMENT_NAME = "Model_Verification_Production" 
MODEL_NAME = "Final_Voting_Ensemble_PROD_PROTO" 
MLFLOW_URI = "sqlite:///C:/Users/szymo/Desktop/house_pricing-20251207T112429Z-3-001/house_pricing/src/mlflow.db"



def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    y_log = np.log1p(y)
    X_train, X_test, y_train, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test_log

def load_final_model():
    model_uri = "models:/Final_Voting_Ensemble_PROD_PROTO/latest"
    return mlflow.pyfunc.load_model(model_uri)

def run_mlops_verification():
    print(f"--- ROZPOCZĘCIE WERYFIKACJI MLFLOW ({EXPERIMENT_NAME})")
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"Verification_of_{MODEL_NAME}") as run:
        X_train, X_test, y_train, y_test_log = load_data()
        final_pipeline = load_final_model() 
        print("Model załadowany PRAWIDŁOWO!")
        y_pred_original = final_pipeline.predict(X_test) 
        y_test_original = np.expm1(y_test_log)
        try:
            rmse_original = root_mean_squared_error(y_test_original, y_pred_original)
        except NameError: 
            rmse_original = mean_squared_error(y_test_original, y_pred_original, squared=False)
            
        print(f"RMSE (Skala oryginalna, metryka biznesowa): {rmse_original:.2f}")
        
        mlflow.log_metric("validation_rmse_original_scale", rmse_original) 
        mlflow.set_tag("status", "VERIFIED")
        mlflow.set_tag("source_model", MODEL_NAME)
        
        print(f"Model ({MODEL_NAME}) zweryfikowany. RMSE: {rmse_original:.2f}")

if __name__ == "__main__":
    run_mlops_verification()