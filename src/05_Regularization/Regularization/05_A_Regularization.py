#!/usr/bin/env python
"""
Regularization.

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


def _ensure_output_dirs() -> None:
    """Crea directorios requeridos de salida."""
    REGULARIZATION_FILES_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    REGULARIZATION_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carga datos de entrenamiento y prueba."""
    path_train = (
        SRC_DIR / "01_Preprocessing" / "Preprocessing"
        / "T_train_final_objetivo.csv"
    )
    path_test = (
        SRC_DIR / "01_Preprocessing" / "Preprocessing"
        / "T_test_final_objetivo.csv"
    )

    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    X_train = df_train.drop(columns=["objetivo"])
    y_train = df_train["objetivo"]
    X_test = df_test.drop(columns=["objetivo"])
    y_test = df_test["objetivo"]
    return X_train, y_train, X_test, y_test


def calcular_metricas(
    modelo: LinearRegression | RidgeCV | LassoCV | ElasticNetCV,
    X: pd.DataFrame,
    y: pd.Series,
    nombre: str,
) -> dict[str, str | float]:
    """Calcula R², RMSE y MAE."""
    y_pred = modelo.predict(X)
    return {
        "Modelo": nombre,
        "R²": r2_score(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE": mean_absolute_error(y, y_pred),
    }


def _train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[LinearRegression, RidgeCV, LassoCV, ElasticNetCV]:
    """Entrena modelos base y regularizados."""
    alphas = np.logspace(-4, 4, 50)
    print("Entrenando modelos con validacion cruzada (5-fold)...\n")

    reg_lineal = LinearRegression()
    reg_lineal.fit(X_train, y_train)
    print("Regresion Lineal base entrenada")

    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train, y_train)
    print(f"Ridge - Mejor alpha: {ridge.alpha_:.6f}")

    lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    print(f"Lasso - Mejor alpha: {lasso.alpha_:.6f}")

    elastic = ElasticNetCV(
        alphas=alphas,
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
        cv=5,
        max_iter=10000,
        random_state=42,
    )
    elastic.fit(X_train, y_train)
    print(
        "Elastic Net - "
        f"Mejor alpha: {elastic.alpha_:.6f}, "
        f"l1_ratio: {elastic.l1_ratio_:.2f}"
    )

    return reg_lineal, ridge, lasso, elastic


def _build_metrics_table(
    models: tuple[LinearRegression, RidgeCV, LassoCV, ElasticNetCV],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Calcula tabla de metricas para todos los modelos."""
    reg_lineal, ridge, lasso, elastic = models
    metricas = [
        calcular_metricas(reg_lineal, X_test, y_test, BASE_MODEL_NAME),
        calcular_metricas(ridge, X_test, y_test, RIDGE_MODEL_NAME),
        calcular_metricas(lasso, X_test, y_test, LASSO_MODEL_NAME),
        calcular_metricas(elastic, X_test, y_test, ELASTIC_MODEL_NAME),
    ]
    return pd.DataFrame(metricas).set_index("Modelo").round(4)


def _save_and_print_metrics(df_metricas: pd.DataFrame) -> None:
    """Imprime y guarda la comparacion de metricas."""
    print("\n" + "=" * 60)
    print("COMPARACION: REGRESION LINEAL vs MODELOS REGULARIZADOS")
    print("=" * 60)
    print(df_metricas.to_string())
    print("=" * 60)

    comparacion_path = FILES_DIR / "comparacion_regularizacion.csv"
    df_metricas.to_csv(comparacion_path)
    print(f"\nGuardado: {comparacion_path}")


def _build_and_save_coefficients(
    feature_names: list[str],
    models: tuple[LinearRegression, RidgeCV, LassoCV, ElasticNetCV],
) -> None:
    """Construye y guarda tabla de coeficientes de modelos."""
    reg_lineal, ridge, lasso, elastic = models

    coeficientes = pd.DataFrame(
        {
            "Feature": feature_names,
            "Reg_Lineal": reg_lineal.coef_,
            "Ridge": ridge.coef_,
            "Lasso": lasso.coef_,
            "ElasticNet": elastic.coef_,
        }
    )

    interceptos = pd.DataFrame(
        {
            "Feature": ["Intercept"],
            "Reg_Lineal": [reg_lineal.intercept_],
            "Ridge": [ridge.intercept_],
            "Lasso": [lasso.intercept_],
            "ElasticNet": [elastic.intercept_],
        }
    )

    coeficientes = pd.concat(
        [interceptos, coeficientes],
        ignore_index=True,
    )
    coeficientes_path = FILES_DIR / "coeficientes_modelos.csv"
    coeficientes.to_csv(coeficientes_path, index=False)

    print("Coeficientes de los modelos:")
    print(coeficientes.round(4).to_string(index=False))
    print(f"\nGuardado: {coeficientes_path}")


def _build_and_save_pca_betas(
    feature_names: list[str],
    reg_lineal: LinearRegression,
    elastic: ElasticNetCV,
) -> None:
    """Guarda comparativa beta pre y post regularizacion para columnas PCA."""
    index_map = {name: idx for idx, name in enumerate(feature_names)}
    pca_cols = [col for col in feature_names if col.startswith("PC")]
    beta_pre = [reg_lineal.coef_[index_map[col]] for col in pca_cols]
    beta_post = [elastic.coef_[index_map[col]] for col in pca_cols]

    betas_pca = pd.DataFrame(
        {
            "Feature": pca_cols,
            "Beta_Pre_Reg (Lineal)": beta_pre,
            "Beta_Post_Reg (ElasticNet)": beta_post,
        }
    )

    betas_path = FILES_DIR / "betas_post_pre_numericas.csv"
    betas_pca.to_csv(betas_path, index=False)
    print("Betas pre/post regularizacion (numericas):")
    print(betas_pca.round(4).to_string(index=False))
    print(f"\nGuardado: {betas_path}")


def _plot_and_save_metrics(df_metricas: pd.DataFrame) -> None:
    """Genera grafico de barras de metricas y lo guarda a disco."""
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
    plt.close()
    print(f"Grafico guardado: {fig_path}")


def _select_best_regularized(
    df_metricas: pd.DataFrame,
) -> tuple[str, float]:
    """Selecciona el mejor modelo regularizado por R²."""
    df_regularizados = df_metricas.drop(BASE_MODEL_NAME)
    best_name = str(df_regularizados["R²"].idxmax())
    best_r2 = float(df_regularizados.loc[best_name, "R²"])
    return best_name, best_r2


def _save_best_bundle(
    best_name: str,
    best_r2: float,
    models: tuple[LinearRegression, RidgeCV, LassoCV, ElasticNetCV],
    feature_names: list[str],
    df_metricas: pd.DataFrame,
) -> None:
    """Guarda modelo ganador, metadata y zip de artefactos."""
    _, ridge, lasso, elastic = models
    modelos_dict = {
        RIDGE_MODEL_NAME: ridge,
        LASSO_MODEL_NAME: lasso,
        ELASTIC_MODEL_NAME: elastic,
    }
    best_model = modelos_dict[best_name]

    bundle_dir = REGULARIZATION_BUNDLE_DIR / "reg_lin_ganador_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    modelo_filename = "modelo_elasticnet.pkl"
    model_path = bundle_dir / modelo_filename
    metadata_path = bundle_dir / METADATA_FILENAME

    joblib.dump(best_model, model_path)

    metadata = {
        "modelo": best_name,
        "alpha": (
            float(best_model.alpha_)
            if hasattr(best_model, "alpha_")
            else None
        ),
        "l1_ratio": (
            float(best_model.l1_ratio_)
            if hasattr(best_model, "l1_ratio_")
            else None
        ),
        "r2_score": best_r2,
        "rmse": float(df_metricas.loc[best_name, "RMSE"]),
        "mae": float(df_metricas.loc[best_name, "MAE"]),
        "features": feature_names,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    zip_path = REGULARIZATION_BUNDLE_DIR / "reg_lin_ganador_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(model_path, model_path.name)
        zipf.write(metadata_path, metadata_path.name)

    print(f"Modelo guardado: {model_path}")
    print(f"Metadata guardada: {metadata_path}")
    print(f"ZIP creado: {zip_path}")


def _print_conclusion(
    df_metricas: pd.DataFrame,
    best_name: str,
) -> None:
    """Imprime resumen final de rendimiento de modelos."""
    r2_base = float(df_metricas.loc[BASE_MODEL_NAME, "R²"])
    r2_ridge = float(df_metricas.loc[RIDGE_MODEL_NAME, "R²"])
    r2_lasso = float(df_metricas.loc[LASSO_MODEL_NAME, "R²"])
    r2_elastic = float(df_metricas.loc[ELASTIC_MODEL_NAME, "R²"])

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"\nRegresion Lineal base: R² = {r2_base:.4f}")
    print(
        "Ridge (L2): "
        f"R² = {r2_ridge:.4f} "
        f"(diferencia: {(r2_ridge - r2_base) * 100:+.2f}%)"
    )
    print(
        "Lasso (L1): "
        f"R² = {r2_lasso:.4f} "
        f"(diferencia: {(r2_lasso - r2_base) * 100:+.2f}%)"
    )
    print(
        "Elastic Net: "
        f"R² = {r2_elastic:.4f} "
        f"(diferencia: {(r2_elastic - r2_base) * 100:+.2f}%)"
    )
    print(f"\n>>> Mejor modelo regularizado: {best_name} <<<")
    print("=" * 60)


def regularization() -> bool:
    """Ejecuta el proceso de regularizacion y retorna True/False."""
    progress = tqdm(total=7, desc="Regularization", unit="paso")
    try:
        _ensure_output_dirs()

        progress.set_postfix_str("Cargando train/test")
        X_train, y_train, X_test, y_test = _load_data()
        progress.update(1)

        feature_names = X_train.columns.tolist()
        print(
            "Datos de entrenamiento: "
            f"{X_train.shape[0]} muestras, "
            f"{X_train.shape[1]} features"
        )
        print(f"Datos de test: {X_test.shape[0]} muestras")

        progress.set_postfix_str("Entrenando modelos")
        models = _train_models(X_train, y_train)
        progress.update(1)

        progress.set_postfix_str("Calculando metricas")
        df_metricas = _build_metrics_table(models, X_test, y_test)
        _save_and_print_metrics(df_metricas)
        progress.update(1)

        progress.set_postfix_str("Guardando reportes")
        _build_and_save_coefficients(feature_names, models)
        reg_lineal, _, _, elastic = models
        _build_and_save_pca_betas(feature_names, reg_lineal, elastic)
        _plot_and_save_metrics(df_metricas)
        progress.update(1)

        progress.set_postfix_str("Empaquetando modelo")
        best_name, best_r2 = _select_best_regularized(df_metricas)
        print(
            "\n>>> MEJOR MODELO REGULARIZADO: "
            f"{best_name} (R² = {best_r2:.4f}) <<<"
        )
        _save_best_bundle(
            best_name,
            best_r2,
            models,
            feature_names,
            df_metricas,
        )
        progress.update(1)

        progress.set_postfix_str("Completado")
        _print_conclusion(df_metricas, best_name)
        progress.update(2)
        return True
    except (KeyError, OSError, pd.errors.ParserError) as error:
        print(f"Error in regularization: {error}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    regularization()
