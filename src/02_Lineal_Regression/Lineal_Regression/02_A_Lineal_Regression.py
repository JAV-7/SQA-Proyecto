"""
Lineal Regression.

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

V 0.2
"""

import json
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = Path(__file__).resolve().parent
MODULE_DIR = TRAINING_DIR.parent
PREPROCESSING_DIR = SRC_DIR / "01_Preprocessing" / "Preprocessing"
MODEL_DIR = MODULE_DIR / "regression_lineal"
REPORTS_DIR = SRC_DIR / "reports"
GRAPHICS_DIR = SRC_DIR / "graphics"
FILES_DIR = SRC_DIR / "files"
LINEAR_FILES_DIR = FILES_DIR / "lineal_regression"

INPUT_DEFAULT_PATH = FILES_DIR / "retail_store_inventory_entrenamiento.csv"
REPORT_PATH = REPORTS_DIR / "01_a_preprocessing_report.txt"


def _ensure_output_dirs() -> None:
    """Crea los directorios de salida si no existen."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LINEAR_FILES_DIR.mkdir(parents=True, exist_ok=True)


SEP = "___"  # Con esto encuentra las columnas Categoricas (One Hot)


def is_binary_series(s: pd.Series) -> bool:
    """Determina si una serie es binaria (0/1) considerando NaNs."""
    vals = pd.unique(s.dropna())
    return set(vals).issubset({0, 1}) or set(vals).issubset({0.0, 1.0})


def prefix_of(col: str, sep: str = SEP) -> str | None:
    """Dado un nombre de columna, devuelve el prefijo antes del separador."""
    return col.split(sep, 1)[0] if sep in col else None


def build_nominal_blocks_by_prefix(
        X: pd.DataFrame, sep: str = SEP
        ) -> dict[str, list[str]]:
    """Agrupa columnas binarias por su prefijo común antes del separador."""
    blocks = {}
    for c in X.columns:
        if sep in c and is_binary_series(X[c]):
            blocks.setdefault(prefix_of(c, sep), []).append(c)
    # respeta orden del CSV
    for k, v in blocks.items():
        blocks[k] = [c for c in X.columns if c in set(v)]
    return blocks


def _load_and_prepare_data(
        progress: tqdm) -> (tuple[pd.DataFrame, pd.DataFrame,
                            np.ndarray, np.ndarray] | None):
    """Carga y prepara datos de entrenamiento y prueba."""
    progress.set_postfix_str("Cargando train/test")
    train_df = pd.read_csv(PREPROCESSING_DIR / "T_train_final_objetivo.csv")
    test_df = pd.read_csv(PREPROCESSING_DIR / "T_test_final_objetivo.csv")
    progress.update(1)

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1].to_numpy(dtype=float)
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1].to_numpy(dtype=float)

    return X_train, y_train, X_test, y_test


def _build_and_train_pipeline(X_train: pd.DataFrame, y_train: np.ndarray,
                              progress: tqdm) -> Pipeline:
    """Construye y entrena el pipeline de regresión lineal."""
    blocks = build_nominal_blocks_by_prefix(X_train, SEP)
    cols_len = 2
    drop_cols = [cols[0]
                 for cols in blocks.values() if len(cols) >= cols_len]

    progress.set_postfix_str("Preparando pipeline")
    progress.update(1)

    arreglar_despeje = ColumnTransformer(
        transformers=[("drop_nominal_bases", "drop", drop_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
        force_int_remainder_cols=False
    )

    modelo = Pipeline([
        ("dropper", arreglar_despeje),
        ("linreg", LinearRegression(fit_intercept=True)),
    ], memory=None)

    modelo.fit(X_train, y_train)
    progress.set_postfix_str("Entrenando modelo")
    progress.update(1)

    return modelo


def _print_model_info(model: Pipeline, X_train: pd.DataFrame) -> None:
    """Imprime información del modelo (intercepto y coeficientes)."""
    intercepto = model.named_steps["linreg"].intercept_
    coefs = model.named_steps["linreg"].coef_
    feature_names = model.named_steps["dropper"].get_feature_names_out(
        X_train.columns)

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    print("Intercepto (beta0):", intercepto)
    print(coef_df)


def _save_artifacts(model: Pipeline, X_train: pd.DataFrame,
                    progress: tqdm) -> None:
    """Guarda el modelo, columnas esperadas y crea archivo ZIP."""
    model_path = MODEL_DIR / "modelo_reg_lineal.pkl"
    expected_columns_path = LINEAR_FILES_DIR / "expected_columns.json"

    joblib.dump(model, model_path)
    expected_cols = X_train.columns.tolist()
    with open(expected_columns_path, "w", encoding="utf-8") as f:
        json.dump(
            {"columns": expected_cols,
             "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            f)

    print("Artefactos guardados:", model_path, expected_columns_path)

    zip_path = MODEL_DIR / "mi_reg_lin_artifacts_bundle.zip"
    candidates = [model_path, expected_columns_path]
    present = [f for f in candidates if Path(f).exists()]

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in present:
            zf.write(f, arcname=Path(f).name)

    print("ZIP creado en:", zip_path)
    print("Incluidos:", present)
    progress.set_postfix_str("Guardando artefactos")
    progress.update(1)


def _evaluate_and_print_metrics(model: Pipeline, X_train: pd.DataFrame,
                                y_train: np.ndarray, X_test: pd.DataFrame,
                                y_test: np.ndarray) -> None:
    """Evalúa el modelo e imprime métricas."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("\n=== MÉTRICAS DE EVALUACIÓN ===")
    print("\nTRAIN:")
    print(f"  R² Score: {r2_score(y_train, y_train_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")

    print("\nTEST:")
    print(f"  R² Score: {r2_score(y_test, y_test_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")


def lineal_regression() -> bool:
    """Ejecuta el entrenamiento de regresión lineal y retorna True/False."""
    progress = tqdm(total=6, desc="Lineal Regression", unit="paso")
    try:
        _ensure_output_dirs()

        X_train, y_train, X_test, y_test = _load_and_prepare_data(progress)
        modelo = _build_and_train_pipeline(X_train, y_train, progress)
        _print_model_info(modelo, X_train)
        _save_artifacts(modelo, X_train, progress)
        _evaluate_and_print_metrics(modelo, X_train, y_train, X_test, y_test)

        progress.set_postfix_str("Completado")
        progress.update(2)

        return True
    except FileNotFoundError as fnf_error:
        print(f"Archivo no encontrado: {fnf_error}")
        return False
    except ValueError as val_error:
        print(f"Error de valor: {val_error}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    lineal_regression()
