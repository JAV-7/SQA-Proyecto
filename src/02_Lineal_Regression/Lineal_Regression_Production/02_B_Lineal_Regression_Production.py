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
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parents[1]
FILES_DIR = SRC_DIR / "files"
MODEL_DIR = MODULE_DIR / "regression_lineal"
LINEAR_FILES_DIR = FILES_DIR / "lineal_regression"
INPUT_CSV_PATH = FILES_DIR / "T_new_final.csv"
OUTPUT_CSV_PATH = FILES_DIR / "Regresion_lineal_nuevos_predicciones.csv"


def lineal_regression_production() -> bool:
    """Ejecuta la regresión lineal de producción y retorna True/False."""
    progress = tqdm(total=4, desc="Lineal Reg Prod", unit="paso")
    try:
        # Cargar modelo y columnas esperadas
        progress.set_postfix_str("Cargando artefactos")
        modelo = joblib.load(MODEL_DIR / "modelo_reg_lineal.pkl")
        with open(LINEAR_FILES_DIR / "expected_columns.json", "r", encoding="utf-8") as f:
            expected_cols = json.load(f)["columns"]
        progress.update(1)

        # Cargar nuevos datos preprocesados
        progress.set_postfix_str("Leyendo datos nuevos")
        df_nuevo = pd.read_csv(INPUT_CSV_PATH)
        progress.update(1)

        # Verificar que las columnas esperadas estén presentes
        if not all(col in df_nuevo.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in df_nuevo.columns]
            print(f"Error: Faltan columnas esperadas en los nuevos datos: {missing_cols}")
            return False

        # Predecir con el modelo
        progress.set_postfix_str("Generando predicciones")
        y_pred = modelo.predict(df_nuevo[expected_cols])
        progress.update(1)

        # Guardar resultados
        out = df_nuevo.copy()
        out["yhat"] = y_pred
        out.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"✅ Predicciones guardadas en: {OUTPUT_CSV_PATH}")
        progress.set_postfix_str("Completado")
        progress.update(1)
        return True

    except Exception as error:
        print(f"Error en regresión lineal de producción: {error}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    lineal_regression_production()

