#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference_api.py: API para servir predicciones del modelo de precios de casas utilizando FastAPI.
Carga los artefactos (modelo, encoders y scaler) y define un endpoint para recibir datos en JSON
y retornar la predicción.
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import wandb

API_PORT = int(os.environ.get("API_PORT", 8000))


# Definir el esquema de datos para la petición de inferencia
class FraudData(BaseModel):
    Payment_6804: float
    Infraction_EJZ: float
    Base_76065: float
    Infraction_GGO: float
    Infraction_TLPJ: float
    Infraction_RKTA: float
    Infraction_CZE: float
    Base_02683: float
    Infraction_BSU: float
    Infraction_FMXQ: float
    Infraction_TBP: float
    Expenditure_JIG: float
    Base_7744: float
    Risk_8902: float
    Infraction_AYWV: float

# Función para transformar variables categóricas usando los encoders guardados
def transform_new_data_ohe(df, imputer):
    """
    Transforma las columnas categóricas de nuevos datos usando los encoders ya ajustados.
    Elimina las columnas originales y concatena las columnas dummy generadas.
    """
    columnas_train = [
    'Payment_6804',
    'Infraction_EJZ',
    'Base_76065',
    'Infraction_GGO',
    'Infraction_TLPJ',
    'Infraction_RKTA',
    'Infraction_CZE',
    'Base_02683',
    'Infraction_BSU',
    'Infraction_FMXQ',
    'Infraction_TBP',
    'Expenditure_JIG',
    'Base_7744',
    'Risk_8902',
    'Infraction_AYWV']

    df_transformed = df.copy()

    df_transformed = pd.DataFrame(imputer.transform(df[columnas_train]), columns=columnas_train)

    return df_transformed

# Inicializar wandb (no inicia un run de tracking)
api = wandb.Api()


# En tu código (antes de usar wandb), añade:
import os
print("=== ENVIRONMENT VARIABLES ===")
print("WANDB_ENTITY:", os.getenv("WANDB_ENTITY"))  # Debe mostrar tu entidad
print("WANDB_API_KEY:", "***" if os.getenv("WANDB_API_KEY") else "No encontrada")  # Solo confirma existencia
print("WEBSITES_PORT:", os.getenv("WEBSITES_PORT"))  # Debe ser 8080

# Definir el nombre del artefacto y usuario/proyecto
artifact_path = os.getenv("WANDB_ARTIFACT_PATH", "sofiaperezperez22-universidad-polit-cnica-de-madrid/Fraud-classification/fraud_classification_model:latest")

# Descargar artefacto
artifact = api.artifact(artifact_path, type="model")
artifact_dir = artifact.download()

# Cargar artefactos guardados
model = joblib.load(os.path.join(artifact_dir, "model.pkl"))
imputer = joblib.load(os.path.join(artifact_dir, "imputer.pkl"))

# Inicializar la aplicación FastAPI
app = FastAPI(title="API de Inferencia - Detección de fraude")

@app.post("/predict_fraud")
async def predict_fraud(data: FraudData):
    """
    Endpoint para recibir datos de una casa y retornar la predicción del precio.
    """
    # Convertir el objeto recibido a DataFrame con model_dump
    input_data = pd.DataFrame([data.model_dump()])

    # Transformar variables categóricas usando los encoders guardados
    input_data = transform_new_data_ohe(input_data, imputer)

    # Realizar la predicción
    pred = model.predict(input_data)

    # Retornar la predicción
    return {"predicted_fraud": float(pred[0])}

# Ejecutar la API (ejecutar este script directamente)
if __name__ == "__main__":
    # Ejecutar la API en el puerto especificado
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)

