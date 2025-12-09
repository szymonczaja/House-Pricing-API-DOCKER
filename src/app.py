import pandas as pd
import mlflow.sklearn 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os, logging, sys
from contextlib import asynccontextmanager

from sklearn.pipeline import Pipeline 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fe_transformer import FeatureTransformer
from median_transformer import GroupedMedianTransformer
from zero_transformer import ZeroImputerTransformer

logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'src', 'mlflow.db')
MLFLOW_URI = f'sqlite:///{DB_PATH}'
MODEL_NAME = 'Production_Docker_Model'
STAGE = 'latest'

ml_models = {} 


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_models 
    print("\n" + "="*50)
    print("-- START: Inicjalizacja usług...")
    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker_model")
        print(f"-- Pobieranie modelu ML z folderu: {model_path}...")
        ml_model = mlflow.sklearn.load_model(model_path) 
        ml_models["model"] = ml_model
        print("-- Model ML załadowany pomyślnie!")
        print('-- Predykcji ceny dokonasz na: http://127.0.0.1:8000/docs')
    except Exception as e:
        print(f"!!! KRYTYCZNY BŁĄD ładowania modelu: {e}")

    print("="*50 + "\n")
    yield 
    
    print("-- Zamykanie usług...")
    ml_models.clear()


app = FastAPI(
    title='House Pricing Prediction Api',
    description='API z modelem MLflow do predykcji cen domów.',
    lifespan=lifespan 
)

class HouseInput(BaseModel):
    data: List[Dict[str, Any]]



@app.get('/')
def index():
    return {'message': 'House pricing API is runing!'}

@app.post('/predict')
def predict(input_data: HouseInput):
    if 'model' not in ml_models:
        raise HTTPException(status_code=500, detail='Model nie jest załadowany')
    try:
        df = pd.DataFrame(input_data.data)
        prediction = ml_models["model"].predict(df)
        return {'predcition': prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Prediction failed: {str(e)}')

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'model' in ml_models}