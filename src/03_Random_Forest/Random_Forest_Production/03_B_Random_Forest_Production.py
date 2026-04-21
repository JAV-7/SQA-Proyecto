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
MODEL_DIR = MODULE_DIR / "random_forest"
INPUT_CSV_PATH = FILES_DIR / "T_new_final.csv"
OUTPUT_CSV_PATH = FILES_DIR / "Random_forest_nuevos_predicciones.csv"


def _resolve_model_dir() -> Path | None:
    """Encuentra la carpeta donde existen los artefactos del modelo."""
    candidates = [
        MODULE_DIR / "random_forest",
        MODULE_DIR / "Random_Forest",
    ]

    for candidate in candidates:
        model_path = candidate / "modelo_random_forest.pkl"
        columns_path = candidate / "expected_columns.json"
        if model_path.exists() and columns_path.exists():
            return candidate

    return None

# Cargar artefactos
def random_forest_production()-> bool:
    """Ejecuta la predicción con el modelo de Random
    Forest en producción y retorna True/False."""
    try:
        model_dir = _resolve_model_dir()
        if model_dir is None:
            print("Error: No se encontraron artefactos del modelo en random_forest ni Random_Forest")
            return False

        modelo = joblib.load(model_dir / "modelo_random_forest.pkl")
        with open(model_dir / "expected_columns.json", "r", encoding="utf-8") as f:
            expected_cols = json.load(f)["columns"]

        df_nuevo = pd.read_csv(INPUT_CSV_PATH)

        if not all(col in df_nuevo.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in df_nuevo.columns]
            print(f"Error: Faltan columnas esperadas en los nuevos datos: {missing_cols}")
            return False

        df_nuevo = df_nuevo[expected_cols]
        y_pred = modelo.predict(df_nuevo)

        out = df_nuevo.copy()
        out["yhat"] = y_pred
        OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Predictions saved: {OUTPUT_CSV_PATH}")
        return True
    except Exception as error:
        print(f"Error in random forest production: {error}")
        return False





