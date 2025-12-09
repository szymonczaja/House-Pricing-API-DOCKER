import mlflow.sklearn 
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fe_transformer import FeatureTransformer
from median_transformer import GroupedMedianTransformer
from zero_transformer import ZeroImputerTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_URI = f"sqlite:///{os.path.join(BASE_DIR, 'src', 'mlflow.db')}"
mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_NAME = "Production_Docker_Model"
STAGE = "latest"
EXPORT_PATH = os.path.join(BASE_DIR, "src", "docker_model")

def export():
    print(f"-> Łączenie z MLflow: {MLFLOW_URI}")
    
    if os.path.exists(EXPORT_PATH):
        shutil.rmtree(EXPORT_PATH)
        print("-> Usunięto stary folder docker_model")
        
    print(f"-> Pobieranie modelu '{MODEL_NAME}/{STAGE}'...")
    
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    
    print(f"-> Zapisywanie modelu do folderu: {EXPORT_PATH}...")
    
    mlflow.sklearn.save_model(model, EXPORT_PATH)
    
    print("SUKCES! Model wyeksportowany do src/docker_model")
    print("-> Teraz przebuduj Dockera!")

if __name__ == "__main__":
    export()