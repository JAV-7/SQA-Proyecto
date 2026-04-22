#!/usr/bin/env python
# coding: utf-8
"""
Random Forest

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

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import time
import os
import zipfile

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

SEP = "___"

# Funciones auxiliares
def is_binary_series(s: pd.Series):
    """Determina si una serie es binaria (0/1) considerando NaNs."""
    vals = pd.unique(s.dropna())
    return set(vals).issubset({0, 1}) or set(vals).issubset({0.0, 1.0})


def prefix_of(col: str, sep=SEP):
    """Dado un nombre de columna, devuelve el prefijo antes del separador si existe."""
    return col.split(sep, 1)[0] if sep in col else None


def build_nominal_blocks_by_prefix(X: pd.DataFrame, sep=SEP):
    """Agrupa columnas binarias por su prefijo común antes del separador."""
    blocks = {}
    for c in X.columns:
        if sep in c and is_binary_series(X[c]):
            blocks.setdefault(prefix_of(c, sep), []).append(c)
    # respeta orden del CSV
    for k, v in blocks.items():
        blocks[k] = [c for c in X.columns if c in set(v)]
    return blocks


def random_forest() -> bool:
    """Ejecuta el entrenamiento de Random Forest y retorna True/False."""
    try:
        train_df = pd.read_csv(
            BASE_DIR / "src" / "01_Preprocessing" / "Preprocessing" / "T_train_final_objetivo.csv"
        )
        test_df = pd.read_csv(
            BASE_DIR / "src" / "01_Preprocessing" / "Preprocessing" / "T_test_final_objetivo.csv"
        )

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1].to_numpy(dtype=float)
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1].to_numpy(dtype=float)

        # 1) --- PRECOMPUTA CON TRAIN ---
        blocks = build_nominal_blocks_by_prefix(X_train, SEP)
        drop_cols = [cols[0] for cols in blocks.values() if len(cols) >= 2]  # primera de cada bloque

        # 2) --- PIPELINE CON RANDOM FOREST ---
        arreglar_despeje = ColumnTransformer(
            transformers=[("drop_nominal_bases", "drop", drop_cols)],
            remainder="passthrough",
            verbose_feature_names_out=False,
            force_int_remainder_cols=False
        )

        mi_random_forest = Pipeline([
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
        ])

        # 3) --- FIT & PRED ---
        print("Training random forest")
        mi_random_forest.fit(X_train, y_train)
        print("Training completed")

        feature_importances = mi_random_forest.named_steps["rf"].feature_importances_
        feature_names = mi_random_forest.named_steps["dropper"].get_feature_names_out(X_train.columns)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importances
        }).sort_values("importance", ascending=False)

        print("\nFeatures importance (sorted):")
        print(importance_df)

        print("\nRandom Forest Parameters:")
        print(f"Trees num: {mi_random_forest.named_steps['rf'].n_estimators}")
        print(f"Max depth: {mi_random_forest.named_steps['rf'].max_depth}")
        print(f"Max features: {mi_random_forest.named_steps['rf'].max_features}")

        y_train_pred = mi_random_forest.predict(X_train)
        y_test_pred = mi_random_forest.predict(X_test)

        print("\n=== MÉTRICAS DE EVALUACIÓN ===")
        print("\nTRAIN:")
        print(f"  R² Score: {r2_score(y_train, y_train_pred):.4f}")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
        print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")

        print("\nTEST:")
        print(f"  R² Score: {r2_score(y_test, y_test_pred):.4f}")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
        print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")

        joblib.dump(mi_random_forest, "modelo_random_forest.pkl")

        expected_cols = X_train.columns.tolist()
        with open("expected_columns.json", "w", encoding="utf-8") as f:
            json.dump({"columns": expected_cols, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)

        importance_df.to_csv("feature_importance.csv", index=False)

        print("\nSaved artefacts:")
        print("  - modelo_random_forest.pkl")
        print("  - expected_columns.json")
        print("  - feature_importance.csv")

        dst_dir = r"mi_random_forest"
        os.makedirs(dst_dir, exist_ok=True)
        zip_path = os.path.join(dst_dir, "mi_random_forest_artifacts_bundle.zip")

        candidates = [
            "modelo_random_forest.pkl",
            "expected_columns.json",
            "feature_importance.csv",
        ]
        present = [f for f in candidates if os.path.exists(f)]

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in present:
                zf.write(f, arcname=os.path.basename(f))

        print("\nZIP creado en:", zip_path)
        print("Incluidos:", present)
        return True
    except Exception as error:
        print(f"Error in random forest: {error}")
        return False

if __name__ == "__main__":
    random_forest()
