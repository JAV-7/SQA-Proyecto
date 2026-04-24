#!/usr/bin/env python
"""
Random Forest.

El propósito de este archivo es entrenar un modelo de Random Forest
utilizando datos previamente preprocesados, con el fin de predecir
el valor objetivo. Además, se generan los artefactos necesarios
para su uso en producción y evaluación del modelo.

 - modelo_random_forest.pkl
 - expected_columns.json
 - feature_importance.csv
 - mi_random_forest_artifacts_bundle.zip

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa

V 0.0
"""

import json
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
FILES_DIR = SRC_DIR / "files"
RANDOM_FOREST_FILES_DIR = FILES_DIR / "random_forest"
MODEL_PATH = MODULE_DIR / "modelo_random_forest.pkl"
EXPECTED_COLUMNS_PATH = RANDOM_FOREST_FILES_DIR / "expected_columns.json"
FEATURE_IMPORTANCE_PATH = RANDOM_FOREST_FILES_DIR / "feature_importance.csv"
ARTIFACTS_DIR = MODULE_DIR / "mi_random_forest"

SEP = "___"


def is_binary_series(s: pd.Series) -> bool:
    """Determina si una serie es binaria (0/1) considerando NaNs."""
    vals = pd.unique(s.dropna())
    return set(vals).issubset({0, 1}) or set(vals).issubset({0.0, 1.0})


def prefix_of(col: str, sep: str = SEP) -> str | None:
    """Dado un nombre de columna, devuelve el prefijo antes del separador."""
    return col.split(sep, 1)[0] if sep in col else None


def build_nominal_blocks_by_prefix(
        X: pd.DataFrame, sep: str = SEP) -> dict[str, list[str]]:
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
        progress: tqdm
        ) -> (tuple[pd.DataFrame, pd.DataFrame,
                    np.ndarray, np.ndarray] | None):
    """Carga y prepara datos de entrenamiento y prueba."""
    RANDOM_FOREST_FILES_DIR.mkdir(parents=True, exist_ok=True)
    progress.set_postfix_str("Cargando train/test")

    train_path = (SRC_DIR / "01_Preprocessing" / "Preprocessing" /
                  "T_train_final_objetivo.csv")
    test_path = (SRC_DIR / "01_Preprocessing" / "Preprocessing" /
                 "T_test_final_objetivo.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1].to_numpy(dtype=float)
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1].to_numpy(dtype=float)

    progress.update(1)
    return X_train, y_train, X_test, y_test


def _build_and_train_pipeline(X_train: pd.DataFrame, y_train: np.ndarray,
                              progress: tqdm) -> Pipeline:
    """Construye y entrena el pipeline de Random Forest."""
    blocks = build_nominal_blocks_by_prefix(X_train, SEP)
    limit_cols = 2
    drop_cols = [cols[0] for cols in blocks.values()
                 if len(cols) >= limit_cols]

    progress.set_postfix_str("Preparando pipeline")
    progress.update(1)

    arreglar_despeje = ColumnTransformer(
        transformers=[("drop_nominal_bases", "drop", drop_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
        force_int_remainder_cols=False
    )

    mi_random_forest = Pipeline(
        [
            ("dropper", arreglar_despeje),
            ("rf", RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=20,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )),
        ],
        memory=None,
    )

    print("Training random forest")
    mi_random_forest.fit(X_train, y_train)
    print("Training completed")
    progress.set_postfix_str("Entrenando modelo")
    progress.update(1)

    return mi_random_forest


def _evaluate_and_print_metrics(model: Pipeline, X_train: pd.DataFrame,
                                y_train: np.ndarray, X_test: pd.DataFrame,
                                y_test: np.ndarray) -> pd.DataFrame:
    """Evalúa el modelo e imprime métricas."""
    feature_importances = (model.named_steps["rf"]
                          .feature_importances_)
    feature_names = model.named_steps["dropper"].get_feature_names_out(
        X_train.columns)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances
    }).sort_values("importance", ascending=False)

    print("\nFeatures importance (sorted):")
    print(importance_df)

    print("\nRandom Forest Parameters:")
    print(f"Trees num: {model.named_steps['rf'].n_estimators}")
    print(f"Max depth: {model.named_steps['rf'].max_depth}")
    print(f"Max features: {model.named_steps['rf'].max_features}")

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

    return importance_df


def _save_artifacts(model: Pipeline, importance_df: pd.DataFrame,
                    X_train: pd.DataFrame, progress: tqdm) -> None:
    """Guarda el modelo, columnas esperadas y características."""
    joblib.dump(model, MODEL_PATH)

    expected_cols = X_train.columns.tolist()
    with open(EXPECTED_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "columns": expected_cols,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print("\nSaved artefacts:")
    print("  - modelo_random_forest.pkl")
    print("  - expected_columns.json")
    print("  - feature_importance.csv")
    progress.set_postfix_str("Guardando artefactos")
    progress.update(1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ARTIFACTS_DIR / "mi_random_forest_artifacts_bundle.zip"

    candidates = [
        MODEL_PATH,
        EXPECTED_COLUMNS_PATH,
        FEATURE_IMPORTANCE_PATH,
    ]
    present = [f for f in candidates if Path(f).exists()]

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for f in present:
            zf.write(f, arcname=Path(f).name)

    print("\nZIP creado en:", zip_path)
    print("Incluidos:", present)


def random_forest() -> bool:
    """Ejecuta el entrenamiento de Random Forest y retorna True/False."""
    progress = tqdm(total=6, desc="Random Forest", unit="paso")
    try:
        data = _load_and_prepare_data(progress)
        X_train, y_train, X_test, y_test = data

        model = _build_and_train_pipeline(X_train, y_train, progress)

        importance_df = _evaluate_and_print_metrics(
            model, X_train, y_train, X_test, y_test
        )

        _save_artifacts(model, importance_df, X_train, progress)
        progress.set_postfix_str("Completado")
        progress.update(2)
        return True
    except FileNotFoundError as fnf_error:
        print(f"Archivo no encontrado: {fnf_error}")
        return False
    except pd.errors.EmptyDataError as ede:
        print(f"Error al leer CSV: {ede}")
        return False
    except KeyError as ke:
        print(f"Error de clave: {ke}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    random_forest()
