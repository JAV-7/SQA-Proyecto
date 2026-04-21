"""
Lineal Regression Production

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa           

V 0.0 
"""

from pathlib import Path

import json
import joblib
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parents[1]
FILES_DIR = SRC_DIR / "files"
MODEL_DIR = MODULE_DIR / "regression_lineal"
INPUT_CSV_PATH = FILES_DIR / "T_new_final.csv"
OUTPUT_CSV_PATH = FILES_DIR / "Regresion_lineal_nuevos_predicciones.csv"


def lineal_regression_production() -> bool:
    """Ejecuta la regresión lineal de producción y retorna True/False."""
    try:
        # Cargar modelo y columnas esperadas
        modelo = joblib.load(MODEL_DIR / "modelo_reg_lineal.pkl")
        with open(MODEL_DIR / "expected_columns.json", "r", encoding="utf-8") as f:
            expected_cols = json.load(f)["columns"]

        # Cargar nuevos datos preprocesados
        df_nuevo = pd.read_csv(INPUT_CSV_PATH)

        # Verificar que las columnas esperadas estén presentes
        if not all(col in df_nuevo.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in df_nuevo.columns]
            print(f"Error: Faltan columnas esperadas en los nuevos datos: {missing_cols}")
            return False

        # Predecir con el modelo
        y_pred = modelo.predict(df_nuevo[expected_cols])

        # Guardar resultados
        out = df_nuevo.copy()
        out["yhat"] = y_pred
        out.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"✅ Predicciones guardadas en: {OUTPUT_CSV_PATH}")
        return True

    except Exception as error:
        print(f"Error en regresión lineal de producción: {error}")
        return False



