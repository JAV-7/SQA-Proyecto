#!/usr/bin/env python
# coding: utf-8
"""
Regularization Production

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa

V 0.0
"""

import pandas as pd
import numpy as np
import joblib
import json

# 1. Cargar Modelo y Metadata

# Cargar modelo ganador
modelo = joblib.load('reg_lin_ganador/reg_lin_ganador_bundle/modelo_elasticnet.pkl')

# Cargar metadata
with open('reg_lin_ganador/reg_lin_ganador_bundle/metadata_modelo.json', 'r') as f:
    metadata = json.load(f)

print(f"Modelo cargado: {metadata['modelo']}")
print(f"Alpha: {metadata['alpha']}")
print(f"R² en test: {metadata['r2_score']:.4f}")

# 2. Cargar Nuevos Datos

# Cargar datos nuevos preprocesados
df_new = pd.read_csv('../01_preprocessing_results/preprocessing_production/T_new_final.csv')

# Verificar columnas
expected_cols = metadata['features']
X_new = df_new[expected_cols]

print(f"Datos nuevos: {X_new.shape[0]} muestras, {X_new.shape[1]} features")

# 3. Generar Predicciones

# Predicciones
y_pred = modelo.predict(X_new)

print(f"Predicciones generadas: {len(y_pred)}")
print(f"Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Media: {y_pred.mean():.2f}")

# Crear DataFrame con predicciones
df_predicciones = X_new.copy()
df_predicciones['yhat'] = y_pred

# Guardar
df_predicciones.to_csv('Regresion_ganadora_nuevos_predicciones.csv', index=False)

print("\nPredicciones guardadas en: Regresion_ganadora_nuevos_predicciones.csv")
print("\nPrimeras 5 predicciones:")
print(df_predicciones[['PC1', 'PC2', 'yhat']].head())

