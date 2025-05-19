#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py: Script para entrenar el modelo de regresión de precios de casas.
Realiza el preprocesamiento (OneHotEncoding para variables categóricas y escalado
para variables numéricas), entrena el modelo y guarda los artefactos en la carpeta artifacts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
import wandb
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from category_encoders import TargetEncoder
import wandb
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

wandb.init(
    project=os.getenv("WANDB_PROJECT", "Fraud-classification"),
    name=os.getenv("WANDB_RUN_NAME", "baseline"),
    config={"model": "XGboost"})

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

# Cargar dataset
datos = pd.read_csv('data/data_labels.csv', sep = ',')
test_data = pd.read_csv('data/data_labels.csv', sep = ',')
TARGET  = 'label'

# Separar la clase mayoritaria y minoritaria
df_majority = datos[datos[TARGET] == 0]
df_minority = datos[datos[TARGET] == 1]

# Hacer downsampling de la clase mayoritaria
df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

# Concatenar clase minoritaria con clase mayoritaria reducida
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Tratamiento variables categóricas
columnas_categoricas = df_balanced.select_dtypes(include = 'object').columns
columnas_entrenamiento = df_balanced.columns.drop(["label"])
df_balanced['label'] = df_balanced['label'].astype('int')

# Valores nulos
nulos = {}
for var in df_balanced.columns:
    nulos[var] = df_balanced[var].isna().sum()/len(df_balanced)*100
porc_nulos_borrar = 40

# columnas a eliminar
nulos_borrar = dict(filter(lambda x: x[1] > porc_nulos_borrar, nulos.items()))
variables_borrar = nulos_borrar.keys()

# Eliminamos las columnas con un alto % de nulos
datos.drop(variables_borrar, axis = 1, inplace = True)
columnas_numericas = df_balanced.select_dtypes(include = ['int64', 'float64']).columns

# Crear el imputador KNN con un número de vecinos (por ejemplo, k=2)
imputer = SimpleImputer(strategy='mean')  # Puedes cambiar a 'median' o 'most_frequent'
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(df_balanced[columnas_train]),
    columns=columnas_train
)

test_data = pd.DataFrame(imputer.transform(test_data[columnas_train]), columns=columnas_train)

# Separar características (X) y variable objetivo (y)
cols_datos = X_train_imputed.columns[X_train_imputed.columns != TARGET]

# División de datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X_train_imputed[cols_datos], df_balanced[TARGET], test_size=0.2, stratify=df_balanced[TARGET], random_state=42
)

best_params_gb = {'colsample_bytree':0.7426112300589203,
'learning_rate':0.24950785363900532,
'max_depth':7,
'n_estimators':200,
'subsample':0.9404357846919617}


# Entrenar el modelo de XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model = xgb.XGBClassifier(
    **best_params_gb,       # Tasa de aprendizaje
    use_label_encoder=False,
    eval_metric='mlogloss'
)
# Tenemos que modificar la target porque XGBoost solo trabaja con valores de 0, 1 en la target
# Y nuestra target original tomaba valores de -1 y 1.
y_train_modified = np.where(y_train == -1, 0, y_train)
y_test_modified = np.where(y_test == -1, 0, y_test)
# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train[columnas_train], y_train_modified)

# Hacer predicciones sobre los datos de prueba
y_pred = model.predict(X_test[columnas_train])

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test_modified, y_pred)
print(f'AUC del modelo: {accuracy:.4f}')
print(f'Matriz de confusion: {confusion_matrix(y_test_modified, y_pred)}')


y_pred = model.predict(X_test[columnas_train])
y_proba = model.predict_proba(X_test[columnas_train])
# Métricas
acc = accuracy_score(y_test_modified, y_pred)
precision = precision_score(y_test_modified, y_pred)
recall = recall_score(y_test_modified, y_pred)
f1 = f1_score(y_test_modified, y_pred)
auc = roc_auc_score(y_test_modified, y_proba[:, 1])

print("Métricas de evaluación:")
print("acc:", acc)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)
print("auc:", auc)


# Crear y registrar artefactos con wandb
artifact = wandb.Artifact("fraud_classification_model", type="model")

# Guardar archivos temporalmente
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(imputer, "artifacts/imputer.pkl")

# Añadir archivos al artefacto
artifact.add_file("artifacts/model.pkl")
artifact.add_file("artifacts/imputer.pkl")

# Registrar el artefacto
wandb.log_artifact(artifact)

print("Artefactos registrados en Weights & Biases.")
print("Artefactos guardados en la carpeta artifacts.")