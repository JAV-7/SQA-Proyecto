#!/usr/bin/env python
# coding: utf-8

"""
Random Forest Production

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa           

V 0.0 
"""

# ===== Carga del modelo y predicción en datos nuevos =====
from pathlib import Path

import pandas as pd
import joblib
import json

SRC_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parents[1]
FILES_DIR = SRC_DIR / "files"
INPUT_CSV_PATH = FILES_DIR / "T_new_final.csv"
OUTPUT_CSV_PATH = FILES_DIR / "Random_forest_nuevos_predicciones.csv"


BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = BASE_DIR / "modelo_random_forest.pkl"
COLS_PATH = BASE_DIR / "expected_columns.json"

# Cargar artefactos
def random_forest_production()-> bool:
    """Ejecuta la predicción con el modelo de Random
    Forest en producción y retorna True/False."""
    try:
        if not MODEL_PATH.exists():
            print(f"Error: no existe el modelo en {MODEL_PATH}")
            return False

        modelo = joblib.load(MODEL_PATH)
        with open(COLS_PATH, "r", encoding="utf-8") as f:
            expected_cols = json.load(f)["columns"]
        df_nuevo = pd.read_csv(INPUT_CSV_PATH)

        if not all(col in df_nuevo.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in df_nuevo.columns]
            print(f"Error: Faltan columnas esperadas en los nuevos datos: {missing_cols}")
            return False

        df_nuevo = df_nuevo[expected_cols]
        y_pred = modelo.predict(df_nuevo)

        out = df_nuevo.copy()
        out["prediction"] = y_pred
        OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Predictions saved: {OUTPUT_CSV_PATH}")
        return True
    except Exception as error:
        print(f"Error in random forest production: {error}")
        return False

if __name__ == "__main__":
    random_forest_production()



