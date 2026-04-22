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
import joblib
import json
from pathlib import Path

# 1. Cargar Modelo y Metadata

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def regularization_production() -> bool:
    try:
        bundle_path = PROJECT_ROOT / "reg_lin_ganador" / "reg_lin_ganador_bundle"
        modelo = joblib.load(bundle_path / "modelo_elasticnet.pkl")

        with open(bundle_path / "metadata_modelo.json", "r") as f:
            metadata = json.load(f)

        print(f"Modelo cargado: {metadata['modelo']}")
        print(f"Alpha: {metadata['alpha']}")
        print(f"R² en test: {metadata['r2_score']:.4f}")

        path_new_data = PROJECT_ROOT / "src" / "files" / "T_new_final.csv"
        df_new = pd.read_csv(path_new_data)
        expected_cols = metadata["features"]
        x_new = df_new[expected_cols]

        print(f"Datos nuevos: {x_new.shape[0]} muestras, {x_new.shape[1]} features")

        y_pred = modelo.predict(x_new)

        print(f"Predicciones generadas: {len(y_pred)}")
        print(f"Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Media: {y_pred.mean():.2f}")

        df_predicciones = x_new.copy()
        df_predicciones["yhat"] = y_pred
        df_predicciones.to_csv("Regresion_ganadora_nuevos_predicciones.csv", index=False)

        print("\nPredicciones guardadas en: Regresion_ganadora_nuevos_predicciones.csv")
        print("\nPrimeras 5 predicciones:")
        print(df_predicciones[["PC1", "PC2", "yhat"]].head())
        return True
    except Exception as error:
        print(f"Error in regularization production: {error}")
        return False

if __name__ == "__main__":
    regularization_production()