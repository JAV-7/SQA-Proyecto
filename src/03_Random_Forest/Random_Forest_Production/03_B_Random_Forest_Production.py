#!/usr/bin/env python

"""
Random Forest Production.

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa

V 0.1
"""

# ===== Carga del modelo y predicción en datos nuevos =====
import json
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2]
FILES_DIR = SRC_DIR / "files"
INPUT_CSV_PATH = FILES_DIR / "T_new_final.csv"
OUTPUT_CSV_PATH = FILES_DIR / "Random_forest_nuevos_predicciones.csv"
RANDOM_FOREST_FILES_DIR = FILES_DIR / "random_forest"


BASE_DIR = Path(__file__).resolve().parents[3]
RANDOM_FOREST_DIR = Path(__file__).resolve().parents[1] / "Random_Forest"

MODEL_PATH = RANDOM_FOREST_DIR / "modelo_random_forest.pkl"
COLS_PATH = RANDOM_FOREST_FILES_DIR / "expected_columns.json"


# Cargar artefactos
def random_forest_production() -> bool:
    """
    Arranque de RF produccción.

    Ejecuta la predicción con el modelo de Random
    Forest en producción y retorna True/False.
    """
    progress = tqdm(total=4, desc="Random Forest Prod", unit="paso")
    try:
        progress.set_postfix_str("Cargando artefactos")
        if not MODEL_PATH.exists():
            print(f"Error: no existe el modelo en {MODEL_PATH}")
            return False

        modelo = joblib.load(MODEL_PATH)
        with open(COLS_PATH, encoding="utf-8") as f:
            expected_cols = json.load(f)["columns"]
        df_nuevo = pd.read_csv(INPUT_CSV_PATH)
        progress.update(1)

        if not all(col in df_nuevo.columns for col in expected_cols):
            missing_cols = [
                col for col in expected_cols if col not in df_nuevo.columns]
            print("Error: Faltan columnas esperadas en los nuevos datos:",
                  f" {missing_cols}")
            return False

        df_nuevo = df_nuevo[expected_cols]
        progress.set_postfix_str("Generando predicciones")
        y_pred = modelo.predict(df_nuevo)
        progress.update(1)

        out = df_nuevo.copy()
        out["prediction"] = y_pred
        OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Predictions saved: {OUTPUT_CSV_PATH}")
        progress.set_postfix_str("Completado")
        progress.update(2)
        return True
    except FileNotFoundError as fnf_error:
        print(f"Archivo no encontrado: {fnf_error}")
        return False
    except json.JSONDecodeError as json_error:
        print(f"Error al decodificar JSON: {json_error}")
        return False
    except pd.errors.EmptyDataError as ede:
        print(f"Error al leer CSV: {ede}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    random_forest_production()
