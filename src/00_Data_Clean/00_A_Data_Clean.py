"""
Data Cleaning.

En esta etapa se realiza la limpieza de los datos, que incluye:
- Identificación y manejo de valores faltantes
- Identificación y manejo de valores duplicados
- Identificación y manejo de outliers

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa

V 0.1
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from Common_Functions.Columns import get_columns

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BASE_DIR = SRC_DIR
REPORTS_DIR = BASE_DIR / "reports"
GRAPHICS_DIR = BASE_DIR / "graphics"
FILES_DIR = BASE_DIR / "files"


INITIAL_INSIGHTS_PATH = REPORTS_DIR / "initial_insights.txt"
CLEAN_DATA_PATH = REPORTS_DIR / "clean_data.txt"


def _ensure_output_dirs() -> None:
    """Asegura que los directorios de salida existan, creándolos si no."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)


def get_initial_insights(df: pd.DataFrame) -> bool:
    """
    Genera insights iniciales del dataset.

    Obtenemos insights iniciales del dataset, como dimensiones, cabecera,
    columnas, valores únicos y resumen de faltantes.

    Parametros
    ----------
    df : pd.DataFrame
        El DataFrame del cual se obtendrán los insights.

    Retorna
    ----------
    bool
        True si se obtuvieron los insights correctamente, False si no.

    Eleva
    ----------
    FileNotFoundError
        Si no se encuentra el archivo para escribir los insights.
    Exception
        Cualquier otro error inesperado al escribir en el archivo.
    """
    _ensure_output_dirs()
    try:
        with open(INITIAL_INSIGHTS_PATH, "w", encoding="utf-8") as file:
            file.write("Obteniendo insights iniciales del dataset...\n\n")
            file.write(f"Dimensiones del dataset: {df.shape}\n\n")
            file.write(f"Cabecera del dataset:\n{df.head()}\n\n")
            file.write(f"Columnas del dataset:\n{list(df.columns)}\n\n")

            file.write("Valores únicos de cada columna:\n")
            for col in df.columns:
                file.write(f"{col}:\n")
                file.write(f"{df[col].unique()}\n")
                file.write("-" * 80 + "\n")

            num_cols, cat_cols = get_columns(df)
            file.write("\nColumnas categoricas:\n")
            file.write(f"{cat_cols.tolist()}\n")
            file.write("Columnas numericas:\n")
            file.write(f"{num_cols.tolist()}\n")

            faltantes = df.isna().sum().sort_values(ascending=False)
            porcentaje = (df.isna().mean() * 100).round(2).sort_values(
                ascending=False
            )
            resumen_faltantes = pd.DataFrame(
                {"faltantes": faltantes, "porcentaje_%": porcentaje}
            )
            file.write("\nResumen de faltantes por columna:\n")
            file.write(f"{resumen_faltantes}\n")

        return True
    except FileNotFoundError as fnf_error:
        print(f"Archivo no encontrado al escribir insights: {fnf_error}")
        return False


def _clean_text_columns(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Limpia columnas de texto: espacios y convirtiendo a minúsculas."""
    object_cols = df_clean.select_dtypes(include="object").columns
    for col in object_cols:
        df_clean[col] = df_clean[col].str.strip().str.lower()
    return df_clean


def _clean_category_column(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Limpia la columna 'Category': caracteres no alfabéticos y errores."""
    if "Category" in df_clean.columns:
        df_clean["Category"] = df_clean["Category"].str.replace(
            r"[^a-z]", "", regex=True
        )
        df_clean["Category"] = df_clean["Category"].replace(
            {
                "electronicsx": "electronics",
                "furniturex": "furniture",
                "clothingx": "clothing",
                "toysx": "toys",
                "groceriesx": "groceries",
            }
        )
    return df_clean


def _plot_numeric_columns(df_clean: pd.DataFrame) -> None:
    """Genera boxplots para columnas numéricas."""
    num_cols, _ = get_columns(df_clean)
    if len(num_cols) == 0:
        return

    fig, axes = plt.subplots(len(num_cols), 1, figsize=(10, 4 * len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]
    for idx, col in enumerate(num_cols):
        sns.boxplot(x=df_clean[col], ax=axes[idx])
        axes[idx].set_title(f"Boxplot - {col}")
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "00_data_clean_boxplots_numericas.png", dpi=150)
    plt.close(fig)


def _plot_categorical_columns(df_clean: pd.DataFrame) -> None:
    """Genera gráficos de frecuencia para columnas categóricas."""
    _, cat_cols = get_columns(df_clean)
    if len(cat_cols) == 0:
        return

    fig, axes = plt.subplots(len(cat_cols), 1, figsize=(12, 4 * len(cat_cols)))
    if len(cat_cols) == 1:
        axes = [axes]
    for idx, col in enumerate(cat_cols):
        sns.countplot(x=df_clean[col], ax=axes[idx], hue=df_clean[col],
                      legend=False)
        axes[idx].set_title(f"Frecuencia - {col}")
        axes[idx].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(GRAPHICS_DIR / "00_data_clean_frecuencias_categoricas.png",
                dpi=150)
    plt.close(fig)


def _write_clean_data_report(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Escribe un reporte detallado del proceso de limpieza de datos."""
    registros_duplicados = df_clean[df_clean.duplicated(keep=False)]
    with open(CLEAN_DATA_PATH, "w", encoding="utf-8") as file:
        file.write("Proceso de limpieza de datos\n")
        file.write("=" * 80 + "\n")
        file.write(f"Dimensiones originales: {df.shape}\n")
        file.write(f"Dimensiones limpias: {df_clean.shape}\n\n")
        file.write("Registros duplicados:\n")
        file.write(f"Total duplicados: {len(registros_duplicados)}\n")
        file.write(f"{registros_duplicados}\n\n")
        file.write("Columnas finales:\n")
        file.write(f"{list(df_clean.columns)}\n")


def clean_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """Limpia el DataFrame y genera reporte y gráficas del proceso."""
    _ensure_output_dirs()

    try:
        df_clean = df.copy()

        if "Weather Condition" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Weather Condition"])

        if "Date" in df_clean.columns:
            df_clean["Date"] = pd.to_datetime(
                df_clean["Date"], dayfirst=True, errors="coerce"
            )
            df_clean["date_timestamp"] = (
                df_clean["Date"].astype("int64", errors="ignore") // 10**9
            )
            df_clean = df_clean.drop(columns=["Date"])

        if "Store ID" in df_clean.columns:
            df_clean["Store ID"] = (
                df_clean["Store ID"].astype(str).str.extract(r"(\d+)")[0]
            )
            df_clean["Store ID"] = pd.to_numeric(
                df_clean["Store ID"], errors="coerce"
            ).astype("Int64")

        if "Product ID" in df_clean.columns:
            df_clean["Product ID"] = (
                df_clean["Product ID"].astype(str).str.extract(r"(\d+)")[0]
            )
            df_clean["Product ID"] = pd.to_numeric(
                df_clean["Product ID"], errors="coerce"
            ).astype("Int64")

        df_clean = _clean_text_columns(df_clean)
        df_clean = _clean_category_column(df_clean)
        _plot_numeric_columns(df_clean)
        _plot_categorical_columns(df_clean)
        _write_clean_data_report(df, df_clean)

        return df_clean
    except FileNotFoundError as fnf_error:
        print(f"Archivo no encontrado durante limpieza de datos: {fnf_error}")
        return None
    except OSError as io_error:
        print(f"Error de IO durante limpieza de datos: {io_error}")
        return None
    except ValueError as val_error:
        print(f"Error de valor durante limpieza de datos: {val_error}")
        return None
    except KeyError as key_error:
        print(f"Error de clave durante limpieza de datos: {key_error}")
        return None
    except TypeError as type_error:
        print(f"Error de tipo durante limpieza de datos: {type_error}")
        return None


def save_clean_data(df: pd.DataFrame) -> bool:
    """Guarda dataset limpio y genera datasets para train y test."""
    _ensure_output_dirs()

    try:
        clean_path = FILES_DIR / "retail_store_inventory_limpio.csv"
        known_prod_path = (
            FILES_DIR / "retail_store_inventory_produccion_known.csv"
        )
        unknown_prod_path = (
            FILES_DIR / "retail_store_inventory_produccion_unknown.csv"
        )
        train_path = FILES_DIR / "retail_store_inventory_entrenamiento.csv"

        df.to_csv(clean_path, index=False)

        sample_size = min(100, len(df))
        df_prod_known = df.sample(sample_size, random_state=42)
        df_prod_known.to_csv(known_prod_path, index=False)

        target_col = "Demand Forecast"
        if target_col in df_prod_known.columns:
            df_prod_unknown = df_prod_known.drop(columns=[target_col])
        else:
            df_prod_unknown = df_prod_known.copy()
        df_prod_unknown.to_csv(unknown_prod_path, index=False)

        df_train = df.drop(df_prod_known.index)
        df_train.to_csv(train_path, index=False)

        return True
    except FileNotFoundError as fnf_error:
        print(f"Archivo no encontrado al guardar datos limpios: {fnf_error}")
        return False
    except OSError as io_error:
        print(f"Error de IO al guardar datos limpios: {io_error}")
        return False


def data_clean() -> bool:
    """Ejecuta el proceso completo de limpieza de datos."""
    progress = tqdm(total=4, desc="Data Clean", unit="paso")
    try:
        progress.set_postfix_str("Cargando dataset")
        df = pd.read_csv(BASE_DIR / "files" / "retail_store_inventory.csv")
        progress.update(1)

        progress.set_postfix_str("Generando insights")
        print("Generando insights...")
        get_initial_insights(df)
        progress.update(1)

        progress.set_postfix_str("Limpiando datos")
        print("Limpiando datos...")
        df_clean = clean_data(df)
        progress.update(1)

        if df_clean is not None:
            progress.set_postfix_str("Guardando salidas")
            print("Guardando datos...")
            save_clean_data(df_clean)
        progress.update(1)

        print("Proceso terminado")
        return True
    except FileNotFoundError as fnf_error:
        print(f"Archivo no encontrado: {fnf_error}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    data_clean()
