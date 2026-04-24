#!/usr/bin/env python
"""
Regularization 

El propósito de este archivo es aplicar técnicas de regularización
(Ridge, Lasso y Elastic Net) sobre modelos de regresión lineal,
con el fin de mejorar el desempeño y reducir el sobreajuste.
 - comparacion_regularizacion.csv
 - coeficientes_modelos.csv
 - betas_post_pre_numericas.csv
 - comparacion_regularizacion.png
 - reg_lin_ganador_bundle.zip

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# 1. Cargar Datos
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
FILES_DIR = SRC_DIR / "files"
GRAPHICS_DIR = SRC_DIR / "graphics"
REGULARIZATION_BUNDLE_DIR = MODULE_DIR / "reg_lin_ganador"
REGULARIZATION_FILES_DIR = FILES_DIR / "regularization"
BASE_MODEL_NAME = "Reg. Lineal (base)"
RIDGE_MODEL_NAME = "Ridge (L2)"
LASSO_MODEL_NAME = "Lasso (L1)"
ELASTIC_MODEL_NAME = "Elastic Net"
METADATA_FILENAME = "metadata_modelo.json"

# 3. Calcular Métricas


def calcular_metricas(modelo, X, y, nombre):
    """Calcula R², RMSE y MAE"""
    y_pred = modelo.predict(X)
    return {
        'Modelo': nombre,
        'R²': r2_score(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred)
    }


def regularization() -> bool:
    """Ejecuta el proceso de regularización y retorna True/False."""
    progress = tqdm(total=7, desc="Regularization", unit="paso")
    try:
        progress.set_postfix_str("Cargando train/test")
        REGULARIZATION_FILES_DIR.mkdir(parents=True, exist_ok=True)
        GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
        path_train = SRC_DIR / "01_Preprocessing" / "Preprocessing" / "T_train_final_objetivo.csv"
        path_test = SRC_DIR / "01_Preprocessing" / "Preprocessing" / "T_test_final_objetivo.csv"

        df_train = pd.read_csv(path_train)
        df_test = pd.read_csv(path_test)

        X_train = df_train.drop(columns=["objetivo"])
        y_train = df_train["objetivo"]
        X_test = df_test.drop(columns=["objetivo"])
        y_test = df_test["objetivo"]
        progress.update(1)

        feature_names = X_train.columns.tolist()

        print(f"Datos de entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} features")
        print(f"Datos de test: {X_test.shape[0]} muestras")

        alphas = np.logspace(-4, 4, 50)
        print("Entrenando modelos con validación cruzada (5-fold)...\n")

        reg_lineal = LinearRegression()
        reg_lineal.fit(X_train, y_train)
        print("Regresión Lineal base entrenada")

        ridge = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(X_train, y_train)
        print(f"Ridge - Mejor alpha: {ridge.alpha_:.6f}")

        lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
        lasso.fit(X_train, y_train)
        print(f"Lasso - Mejor alpha: {lasso.alpha_:.6f}")

        elastic = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95], cv=5, max_iter=10000, random_state=42)
        elastic.fit(X_train, y_train)
        print(f"Elastic Net - Mejor alpha: {elastic.alpha_:.6f}, l1_ratio: {elastic.l1_ratio_:.2f}")
        progress.set_postfix_str("Entrenando modelos")
        progress.update(1)

        metricas = [
            calcular_metricas(reg_lineal, X_test, y_test, BASE_MODEL_NAME),
            calcular_metricas(ridge, X_test, y_test, RIDGE_MODEL_NAME),
            calcular_metricas(lasso, X_test, y_test, LASSO_MODEL_NAME),
            calcular_metricas(elastic, X_test, y_test, ELASTIC_MODEL_NAME)
        ]

        df_metricas = pd.DataFrame(metricas).set_index("Modelo").round(4)

        print("\n" + "=" * 60)
        print("COMPARACIÓN: REGRESIÓN LINEAL vs MODELOS REGULARIZADOS")
        print("=" * 60)
        print(df_metricas.to_string())
        print("=" * 60)

        comparacion_path = FILES_DIR / "comparacion_regularizacion.csv"
        df_metricas.to_csv(comparacion_path)
        print(f"\nGuardado: {comparacion_path}")
        progress.set_postfix_str("Calculando metricas")
        progress.update(1)

        coeficientes = pd.DataFrame({
            "Feature": feature_names,
            "Reg_Lineal": reg_lineal.coef_,
            "Ridge": ridge.coef_,
            "Lasso": lasso.coef_,
            "ElasticNet": elastic.coef_
        })

        interceptos = pd.DataFrame({
            "Feature": ["Intercept"],
            "Reg_Lineal": [reg_lineal.intercept_],
            "Ridge": [ridge.intercept_],
            "Lasso": [lasso.intercept_],
            "ElasticNet": [elastic.intercept_]
        })

        coeficientes = pd.concat([interceptos, coeficientes], ignore_index=True)
        coeficientes_path = FILES_DIR / "coeficientes_modelos.csv"
        coeficientes.to_csv(coeficientes_path, index=False)

        print("Coeficientes de los modelos:")
        print(coeficientes.round(4).to_string(index=False))
        print(f"\nGuardado: {coeficientes_path}")

        pca_cols = [col for col in feature_names if col.startswith("PC")]

        betas_pca = pd.DataFrame({
            "Feature": pca_cols,
            "Beta_Pre_Reg (Lineal)": [reg_lineal.coef_[feature_names.index(col)] for col in pca_cols],
            "Beta_Post_Reg (ElasticNet)": [elastic.coef_[feature_names.index(col)] for col in pca_cols]
        })

        betas_path = FILES_DIR / "betas_post_pre_numericas.csv"
        betas_pca.to_csv(betas_path, index=False)
        print("Betas pre/post regularización (numéricas):")
        print(betas_pca.round(4).to_string(index=False))
        print(f"\nGuardado: {betas_path}")
        progress.set_postfix_str("Guardando reportes")
        progress.update(1)

        _, axes = plt.subplots(1, 3, figsize=(14, 4))

        modelos = df_metricas.index.tolist()
        colores = ["steelblue", "orange", "green", "red"]

        axes[0].bar(modelos, df_metricas["R²"], color=colores)
        axes[0].set_ylabel("R²")
        axes[0].set_title("R² Score (mayor es mejor)")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(modelos, df_metricas["RMSE"], color=colores)
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("RMSE (menor es mejor)")
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(modelos, df_metricas["MAE"], color=colores)
        axes[2].set_ylabel("MAE")
        axes[2].set_title("MAE (menor es mejor)")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        fig_path = GRAPHICS_DIR / "comparacion_regularizacion.png"
        plt.savefig(fig_path, dpi=150)
        plt.show()

        print(f"Gráfico guardado: {fig_path}")

        df_regularizados = df_metricas.drop(BASE_MODEL_NAME)
        mejor_modelo_nombre = df_regularizados["R²"].idxmax()
        mejor_r2 = df_regularizados.loc[mejor_modelo_nombre, "R²"]

        print(f"\n>>> MEJOR MODELO REGULARIZADO: {mejor_modelo_nombre} (R² = {mejor_r2:.4f}) <<<")

        modelos_dict = {
            "Ridge (L2)": ridge,
            "Lasso (L1)": lasso,
            "Elastic Net": elastic
        }
        mejor_modelo = modelos_dict[mejor_modelo_nombre]

        REGULARIZATION_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        bundle_dir = REGULARIZATION_BUNDLE_DIR / "reg_lin_ganador_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        modelo_filename = "modelo_elasticnet.pkl"
        joblib.dump(mejor_modelo, bundle_dir / modelo_filename)

        metadata = {
            "modelo": mejor_modelo_nombre,
            "alpha": float(mejor_modelo.alpha_) if hasattr(mejor_modelo, "alpha_") else None,
            "l1_ratio": float(mejor_modelo.l1_ratio_) if hasattr(mejor_modelo, "l1_ratio_") else None,
            "r2_score": float(mejor_r2),
            "rmse": float(df_regularizados.loc[mejor_modelo_nombre, "RMSE"]),
            "mae": float(df_regularizados.loc[mejor_modelo_nombre, "MAE"]),
            "features": feature_names,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(bundle_dir / METADATA_FILENAME, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Modelo guardado: {bundle_dir / modelo_filename}")
        print(f"Metadata guardada: {bundle_dir / METADATA_FILENAME}")

        zip_path = REGULARIZATION_BUNDLE_DIR / "reg_lin_ganador_bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(bundle_dir / modelo_filename, modelo_filename)
            zipf.write(bundle_dir / METADATA_FILENAME, METADATA_FILENAME)

        print(f"ZIP creado: {zip_path}")
        progress.set_postfix_str("Empaquetando modelo")
        progress.update(1)

        print("\n" + "=" * 60)
        print("CONCLUSIÓN")
        print("=" * 60)

        r2_base = df_metricas.loc[BASE_MODEL_NAME, "R²"]
        r2_ridge = df_metricas.loc[RIDGE_MODEL_NAME, "R²"]
        r2_lasso = df_metricas.loc[LASSO_MODEL_NAME, "R²"]
        r2_elastic = df_metricas.loc[ELASTIC_MODEL_NAME, "R²"]

        print(f"\nRegresión Lineal base: R² = {r2_base:.4f}")
        print(f"Ridge (L2): R² = {r2_ridge:.4f} (diferencia: {(r2_ridge - r2_base) * 100:+.2f}%)")
        print(f"Lasso (L1): R² = {r2_lasso:.4f} (diferencia: {(r2_lasso - r2_base) * 100:+.2f}%)")
        print(f"Elastic Net: R² = {r2_elastic:.4f} (diferencia: {(r2_elastic - r2_base) * 100:+.2f}%)")

        print(f"\n>>> Mejor modelo regularizado: {mejor_modelo_nombre} <<<")
        print("=" * 60)
        progress.set_postfix_str("Completado")
        progress.update(2)
        return True
    except Exception as error:
        print(f"Error in regularization: {error}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    regularization()
