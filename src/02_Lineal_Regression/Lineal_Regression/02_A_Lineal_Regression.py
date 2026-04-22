"""
Lineal Regression

El propòsito de este archivo es entrenar un modelo de regresiòn lineal
para predecir el precio objetivo. Se generan los artefactos necesarios
para la etapa de produccciòn y la posterior evaluaciòn de modelos.

 - T_train_final_objetivo.csv
 - T_test_final_objetivo.csv"
 - pca_pipe_num.joblib
 - pca_metadata.json

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
import os, zipfile
import joblib, json, time
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error




SRC_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = Path(__file__).resolve().parent
MODULE_DIR = TRAINING_DIR.parent
PREPROCESSING_DIR = SRC_DIR / "01_Preprocessing" / "Preprocessing"
MODEL_DIR = MODULE_DIR / "regression_lineal"
REPORTS_DIR = SRC_DIR / "reports"
GRAPHICS_DIR = SRC_DIR / "graphics"
FILES_DIR = SRC_DIR / "files"

INPUT_DEFAULT_PATH = FILES_DIR / "retail_store_inventory_entrenamiento.csv"
REPORT_PATH = REPORTS_DIR / "01_a_preprocessing_report.txt"


def _ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEP = "___"  # Con esto encuentra las columnas Categoricas (One Hot)


def is_binary_series(s: pd.Series):
    vals = pd.unique(s.dropna())
    return set(vals).issubset({0, 1}) or set(vals).issubset({0.0, 1.0})


def prefix_of(col: str, sep=SEP):
    return col.split(sep, 1)[0] if sep in col else None


def build_nominal_blocks_by_prefix(X: pd.DataFrame, sep=SEP):
    blocks = {}
    for c in X.columns:
        if sep in c and is_binary_series(X[c]):
            blocks.setdefault(prefix_of(c, sep), []).append(c)
    # respeta orden del CSV
    for k, v in blocks.items():
        blocks[k] = [c for c in X.columns if c in set(v)]
    return blocks

def lineal_regression()-> bool:
    try:
        _ensure_output_dirs()

        train_df = pd.read_csv(PREPROCESSING_DIR / "T_train_final_objetivo.csv")
        test_df = pd.read_csv(PREPROCESSING_DIR / "T_test_final_objetivo.csv")


        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1].to_numpy(dtype=float)

        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1].to_numpy(dtype=float)
        
        # 1) --- PRECOMPUTA CON TRAIN ---
        blocks = build_nominal_blocks_by_prefix(X_train, SEP)
        drop_cols = [cols[0] for cols in blocks.values() if len(cols) >= 2]  # primera de cada bloque

        # 2) --- PIPELINE SIN CLASES (usa ColumnTransformer para dropear fijo) ---
        arreglar_despeje = ColumnTransformer(
            transformers=[("drop_nominal_bases", "drop", drop_cols)],
            remainder="passthrough",
            verbose_feature_names_out=False,
            force_int_remainder_cols=False
        )

        mi_regresion_lineal = Pipeline([
            ("dropper", arreglar_despeje),
            ("linreg", LinearRegression(fit_intercept=True)),
        ], memory=None)

        # 3) --- FIT & PRED ---
        mi_regresion_lineal.fit(X_train, y_train)


        # Intercepto y coeficientes del modelo dentro del pipeline
        intercepto = mi_regresion_lineal.named_steps["linreg"].intercept_
        coefs = mi_regresion_lineal.named_steps["linreg"].coef_

        # Nombres de columnas después del dropper (lo más directo)
        feature_names = mi_regresion_lineal.named_steps["dropper"].get_feature_names_out(X_train.columns)

        # Mostrar los estimadores beta_0,beta_1,...,beta_p
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        print("Intercepto (beta0):", intercepto)
        print(coef_df)

        # guarda el pipeline completo (dropper + LinearRegression)
        model_path = MODEL_DIR / "modelo_reg_lineal.pkl"
        expected_columns_path = MODEL_DIR / "expected_columns.json"
        joblib.dump(mi_regresion_lineal, model_path)

        # guarda el orden/esperado de columnas de entrenamiento
        expected_cols = X_train.columns.tolist()
        with open(expected_columns_path, "w", encoding="utf-8") as f:
            json.dump({"columns": expected_cols, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)

        print("Artefactos guardados:", model_path, expected_columns_path)


        # Carpeta destino en tu PC
        dst_dir = MODEL_DIR
        os.makedirs(dst_dir, exist_ok=True)
        zip_path = os.path.join(dst_dir, "mi_reg_lin_artifacts_bundle.zip")

        # Archivos que quieres incluir (ajusta si te falta alguno)
        candidates = [
            model_path,
            expected_columns_path,
        ]

        present = [f for f in candidates if Path(f).exists()]
        # Si quieres incluir una carpeta (p. ej., 'sample_data'), descomenta:
        # for root, _, files in os.walk("sample_data"):
        #     for f in files:
        #         present.append(os.path.join(root, f))

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in present:
                zf.write(f, arcname=Path(f).name)  # guarda sin subcarpetas

        print("ZIP creado en:", zip_path)
        print("Incluidos:", present)

        # Evaluación en train y test
        y_train_pred = mi_regresion_lineal.predict(X_train)
        y_test_pred = mi_regresion_lineal.predict(X_test)

        print("\n=== MÉTRICAS DE EVALUACIÓN ===")
        print("\nTRAIN:")
        print(f"  R² Score: {r2_score(y_train, y_train_pred):.4f}")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
        print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")

        print("\nTEST:")
        print(f"  R² Score: {r2_score(y_test, y_test_pred):.4f}")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
        print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")

        return True
    except Exception as error:
        print(f"Error in lineal regression: {error}")
        return False

if __name__ == "__main__":
    lineal_regression()

