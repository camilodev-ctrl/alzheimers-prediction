import gdown 
import os
import joblib

fileId = "1uRygTa97EF3pYkltEe8m_zs84Jk87ayt"
modelPath = "models/alzheimerModel.pkl"

def download_model():
    if not os.path.exists(modelPath):
        print("Descargando el modelo desde Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={fileId}", modelPath, quiet=False)
        print("Modelo descargado exitosamente.")
    else:
        print(" Modelo ya descargado.")


def load_model():
    download_model()
    print(" Cargando el modelo...")
    model = joblib.load(modelPath)
    print("Modelo cargado con éxito.")
    return model