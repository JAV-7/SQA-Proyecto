#!/usr/bin/env python
"""
Model Evaluation.

El propósito de este archivo es comparar el rendimiento de los modelos
de Regresión Lineal y Random Forest utilizando datos de prueba,
con el fin de seleccionar el mejor modelo para producción.

Se calculan métricas como R², RMSE, MAE y MAPE, además de generar
visualizaciones.
 - comparacion_modelos.png
 - metricas_comparativas.csv

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa

V 0.1
"""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "01_Preprocessing" / "Preprocessing"
REPORTS_DIR = PROJECT_ROOT / "src" / "reports"
GRAPHICS_DIR = PROJECT_ROOT / "src" / "graphics"

R2_GREEN_MIN = 0.70
R2_YELLOW_MIN = 0.40
NRMSE_GREEN_MAX = 0.50
NRMSE_YELLOW_MAX = 0.80
MODEL_RF_NAME = "Random Forest"
MODEL_LINEAR_NAME = "Regresion Lineal"


def _ensure_output_dirs() -> None:
    """Crea directorios de salida para reportes y graficas."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)


def calcular_metricas(
    y_real: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    nombre_modelo: str,
) -> dict[str, str | float]:
    """Calcula R², RMSE, MAE y MAPE."""
    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    nrmse = rmse / (np.max(y_real) - np.min(y_real))
    mae = mean_absolute_error(y_real, y_pred)

    denominator = np.where(y_real == 0, 1, y_real)
    mape = np.mean(np.abs((y_real - y_pred) / denominator)) * 100

    return {
        "Modelo": nombre_modelo,
        "R²": r2,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MAE": mae,
        "MAPE (%)": mape,
    }


def _load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carga datos de prueba y separa variables."""
    path_data = PREPROCESSING_DIR / "T_test_final_objetivo.csv"
    df_test = pd.read_csv(path_data)

    X_test = df_test.drop(columns=["objetivo"])
    y_test = df_test["objetivo"]
    return X_test, y_test


def _load_models() -> tuple[Any, Any]:
    """Carga los modelos entrenados desde disco."""
    path_lineal = (
        PROJECT_ROOT
        / "src"
        / "02_Lineal_Regression"
        / "regression_lineal"
        / "modelo_reg_lineal.pkl"
    )
    path_rf = (
        PROJECT_ROOT
        / "src"
        / "03_Random_Forest"
        / "Random_Forest"
        / "modelo_random_forest.pkl"
    )
    return joblib.load(path_lineal), joblib.load(path_rf)


def _print_data_summary(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Muestra resumen de los datos de prueba."""
    print(
        "Datos de test: "
        f"{X_test.shape[0]} muestras, "
        f"{X_test.shape[1]} features"
    )
    print(
        "Variable objetivo - "
        f"Min: {y_test.min():.2f}, "
        f"Max: {y_test.max():.2f}, "
        f"Media: {y_test.mean():.2f}"
    )


def _print_prediction_summary(
    y_pred_lineal: np.ndarray,
    y_pred_rf: np.ndarray,
) -> None:
    """Muestra resumen basico de predicciones por modelo."""
    print(
        "Predicciones Regresion Lineal - "
        f"Min: {y_pred_lineal.min():.2f}, "
        f"Max: {y_pred_lineal.max():.2f}"
    )
    print(
        "Predicciones Random Forest - "
        f"Min: {y_pred_rf.min():.2f}, "
        f"Max: {y_pred_rf.max():.2f}"
    )


def _create_metrics_table(
    metricas_lineal: dict[str, str | float],
    metricas_rf: dict[str, str | float],
) -> pd.DataFrame:
    """Construye tabla comparativa de metricas redondeada."""
    df_metricas = pd.DataFrame([metricas_lineal, metricas_rf])
    df_metricas = df_metricas.set_index("Modelo")
    return df_metricas.round(4)


def _plot_model_comparison(
    y_test: pd.Series,
    y_pred_lineal: np.ndarray,
    y_pred_rf: np.ndarray,
    metricas_lineal: dict[str, str | float],
    metricas_rf: dict[str, str | float],
) -> plt.Figure:
    """Genera grafica comparativa entre modelos."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    y_min, y_max = y_test.min(), y_test.max()

    axes[0, 0].scatter(y_test, y_pred_lineal, alpha=0.3, s=10)
    axes[0, 0].plot([y_min, y_max], [y_min, y_max], "r--", lw=2)
    axes[0, 0].set_xlabel("Valores Reales")
    axes[0, 0].set_ylabel("Predicciones")
    axes[0, 0].set_title(
        "Regresion Lineal "
        f"(R²={float(metricas_lineal['R²']):.4f})"
    )

    axes[0, 1].scatter(y_test, y_pred_rf, alpha=0.3, s=10)
    axes[0, 1].plot([y_min, y_max], [y_min, y_max], "r--", lw=2)
    axes[0, 1].set_xlabel("Valores Reales")
    axes[0, 1].set_ylabel("Predicciones")
    axes[0, 1].set_title(
        "Random Forest "
        f"(R²={float(metricas_rf['R²']):.4f})"
    )

    residuos_lineal = y_test - y_pred_lineal
    axes[1, 0].hist(residuos_lineal, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(x=0, color="r", linestyle="--")
    axes[1, 0].set_xlabel("Residuos")
    axes[1, 0].set_ylabel("Frecuencia")
    axes[1, 0].set_title("Distribucion de Residuos - Reg. Lineal")

    residuos_rf = y_test - y_pred_rf
    axes[1, 1].hist(residuos_rf, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(x=0, color="r", linestyle="--")
    axes[1, 1].set_xlabel("Residuos")
    axes[1, 1].set_ylabel("Frecuencia")
    axes[1, 1].set_title("Distribucion de Residuos - Random Forest")

    fig.tight_layout()
    return fig


def _get_veredicto(metricas: dict[str, str | float]) -> tuple[str, str]:
    """Calcula veredicto de calidad segun R² y NRMSE."""
    r2 = float(metricas["R²"])
    nrmse = float(metricas["NRMSE"])

    is_green = (r2 >= R2_GREEN_MIN) and (nrmse <= NRMSE_GREEN_MAX)
    is_yellow = (R2_YELLOW_MIN <= r2 < R2_GREEN_MIN) or (
        NRMSE_GREEN_MAX < nrmse <= NRMSE_YELLOW_MAX
    )

    if is_green:
        return "VERDE", "confiable para predecir aqui dentro."
    if is_yellow:
        return "AMARILLO", "usable con cautela (depende del caso de uso)."
    return "ROJO", "no confiable para prediccion aqui dentro."


def _print_comparative_report(
    metricas_lineal: dict[str, str | float],
    metricas_rf: dict[str, str | float],
    df_metricas: pd.DataFrame,
) -> None:
    """Imprime la tabla comparativa y conclusiones finales."""
    print("\n" + "=" * 60)
    print("TABLA COMPARATIVA DE MODELOS")
    print("=" * 60)
    print(df_metricas.to_string())
    print("=" * 60)

    mejor_r2 = (
        MODEL_RF_NAME
        if float(metricas_rf["R²"]) > float(metricas_lineal["R²"])
        else MODEL_LINEAR_NAME
    )
    mejor_rmse = (
        MODEL_RF_NAME
        if float(metricas_rf["RMSE"]) < float(metricas_lineal["RMSE"])
        else MODEL_LINEAR_NAME
    )
    mejor_mae = (
        MODEL_RF_NAME
        if float(metricas_rf["MAE"]) < float(metricas_lineal["MAE"])
        else MODEL_LINEAR_NAME
    )

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"\nMejor R²: {mejor_r2}")
    print(f"Mejor RMSE: {mejor_rmse}")
    print(f"Mejor MAE: {mejor_mae}")

    ganador = mejor_r2
    print(f"\n>>> MODELO GANADOR: {ganador} <<<")

    if ganador == MODEL_RF_NAME:
        print(
            "\nJustificacion: Random Forest tiene mejor R² "
            f"({float(metricas_rf['R²']):.4f} vs "
            f"{float(metricas_lineal['R²']):.4f})"
        )
        print(
            "y menor error "
            f"(RMSE: {float(metricas_rf['RMSE']):.2f} vs "
            f"{float(metricas_lineal['RMSE']):.2f})"
        )
    else:
        print(
            "\nJustificacion: Regresion Lineal tiene mejor R² "
            f"({float(metricas_lineal['R²']):.4f} vs "
            f"{float(metricas_rf['R²']):.4f})"
        )

    veredicto_rf, significado_rf = _get_veredicto(metricas_rf)
    print(
        "El veredicto para el modelo Random Forest es: "
        f"{veredicto_rf}\n{significado_rf}"
    )

    veredicto_lineal, significado_lineal = _get_veredicto(metricas_lineal)
    print(
        "El veredicto para el modelo Regresion Lineal es: "
        f"{veredicto_lineal}\n{significado_lineal}"
    )


def _save_outputs(df_metricas: pd.DataFrame, fig: plt.Figure) -> None:
    """Guarda grafica y metricas en archivos de salida."""
    ruta_grafica = GRAPHICS_DIR / "comparacion_modelos.png"
    ruta_csv = REPORTS_DIR / "metricas_comparativas.csv"

    fig.savefig(ruta_grafica, dpi=150)
    df_metricas.to_csv(ruta_csv)
    plt.close(fig)

    print("Resultados guardados en:")
    print(f"- {ruta_grafica}")
    print(f"- {ruta_csv}")


def model_evaluation() -> bool:
    """Ejecuta la evaluación de modelos y retorna True/False."""
    progress = tqdm(total=5, desc="Model Evaluation", unit="paso")
    try:
        _ensure_output_dirs()

        progress.set_postfix_str("Cargando test")
        X_test, y_test = _load_test_data()
        progress.update(1)
        _print_data_summary(X_test, y_test)

        progress.set_postfix_str("Cargando modelos")
        modelo_lineal, modelo_rf = _load_models()
        progress.update(1)

        progress.set_postfix_str("Generando predicciones")
        y_pred_lineal = modelo_lineal.predict(X_test)
        y_pred_rf = modelo_rf.predict(X_test)
        progress.update(1)
        _print_prediction_summary(y_pred_lineal, y_pred_rf)

        metricas_lineal = calcular_metricas(
            y_test, y_pred_lineal, MODEL_LINEAR_NAME
        )
        metricas_rf = calcular_metricas(y_test, y_pred_rf, MODEL_RF_NAME)
        df_metricas = _create_metrics_table(metricas_lineal, metricas_rf)

        fig = _plot_model_comparison(
            y_test,
            y_pred_lineal,
            y_pred_rf,
            metricas_lineal,
            metricas_rf,
        )

        progress.set_postfix_str("Construyendo reporte")
        progress.update(1)

        _print_comparative_report(metricas_lineal, metricas_rf, df_metricas)
        _save_outputs(df_metricas, fig)

        progress.set_postfix_str("Completado")
        progress.update(1)
        return True
    except (FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"Error in model evaluation: {error}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    model_evaluation()
