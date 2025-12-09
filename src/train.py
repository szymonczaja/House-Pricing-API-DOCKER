import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import os
import ast
import warnings
import logging
from pipeline_def import build_final_model_pipeline


SOURCE_EXPERIMENT_NAME = "Final_model_verifaction"
SOURCE_RUN_NAME = "Voted_ensemble_FINAL"

TARGET_EXPERIMENT_NAME = "Production_Training_Pipeline"
REGISTERED_MODEL_NAME = "Production_Docker_Model"

MLFLOW_URI = "sqlite:///C:/Users/szymo/Desktop/house_pricing-20251207T112429Z-3-001/house_pricing/src/mlflow.db"
DATA_PATH = 'src/data/train.csv'
TARGET_COLUMN = 'SalePrice'
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def get_params_from_final_run():
    print(f"> Łączenie z MLflow i szukanie eksperymentu: '{SOURCE_EXPERIMENT_NAME}'...")
    
    experiment = mlflow.get_experiment_by_name(SOURCE_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"!Nie znaleziono eksperymentu '{SOURCE_EXPERIMENT_NAME}'!")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{SOURCE_RUN_NAME}'",
        max_results=1
    )
    
    if runs.empty:
        print(f"!!! Nie znaleziono runu o nazwie '{SOURCE_RUN_NAME}'. Próbuję pobrać NAJNOWSZY run z tego eksperymentu.")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
    if runs.empty:
        raise ValueError("!!!Eksperyment jest pusty! Nie mam skąd pobrać parametrów.")

    best_run = runs.iloc[0]
    print(f"> Znaleziono Run ID: {best_run.run_id}")
    raw_params = best_run.filter(like='params.').to_dict()
    clean_params = {k.replace('params.', ''): v for k, v in raw_params.items()}
    
    return clean_params

def parse_model_params(all_params):
    xgb_params = {'random_state': 42, 'n_jobs': -1}
    ridge_params = {'random_state': 42}
    weights = [0.5, 0.5]

    print("> Parsowanie parametrów...")
    
    for key, value in all_params.items():
        if key.startswith('xgb_'):
            real_key = key.replace('xgb_', '') 
            try:
                if value.replace('.','',1).isdigit():
                    if '.' in value:
                        xgb_params[real_key] = float(value)
                    else:
                        xgb_params[real_key] = int(value)
                else:
                    xgb_params[real_key] = value
            except:
                xgb_params[real_key] = value

        elif key == 'ridge_alpha':
            try:
                ridge_params['alpha'] = float(value)
            except:
                ridge_params['alpha'] = 10.0

        elif key == 'voting_weights':
            try:
                weights = ast.literal_eval(value)
            except:
                print(f"> Nie udało się sparsować wag: {value}. Używam domyślnych.")

    return xgb_params, ridge_params, weights

def load_data():
    if not os.path.exists(DATA_PATH):
        alt_path = os.path.join("data", "train.csv")
        if os.path.exists(alt_path):
             return pd.read_csv(alt_path)
        else:
             raise FileNotFoundError(f"Nie znaleziono pliku: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def run_training():
    print("-- ROZPOCZĘCIE TRENINGU")
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    try:
        raw_params = get_params_from_final_run()
        xgb_params, ridge_params, weights = parse_model_params(raw_params)
        
        print(f"> Pobrane parametry XGB: {xgb_params}")
        print(f"> Pobrane parametry Ridge: {ridge_params}")
        print(f"> Pobrane Wagi Voting: {weights}")
        
    except Exception as e:
        print(f"BŁĄD POBIERANIA PARAMETRÓW: {e}")
        return

    mlflow.set_experiment(TARGET_EXPERIMENT_NAME)
    
    print("--Wczytywanie danych...")
    df = load_data()
    X = df.drop(columns=[TARGET_COLUMN], axis=1, errors='ignore')
    y = df[TARGET_COLUMN] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    
    xgb_model = XGBRegressor(**xgb_params)
    ridge_model = Ridge(**ridge_params)
    
    print("-- Budowanie pipeline'u z plików .py...")
    final_pipeline = build_final_model_pipeline(xgb_model, ridge_model, weights=weights)
    
    print("-- Trenowanie modelu (to może chwilę potrwać)...")
    with mlflow.start_run(run_name="Docker_Production_Build") as run:
        
        final_pipeline.fit(X_train, y_train)
        mlflow.log_params(xgb_params)
        mlflow.log_params(ridge_params)
        mlflow.log_param("weights", str(weights))
        
        print(f"-- Rejestracja modelu '{REGISTERED_MODEL_NAME}'...")
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.iloc[:5]
        )
        print(f"-- SUKCES! Model gotowy. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    run_training()