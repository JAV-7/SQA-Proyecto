"""
Common Functions: Columns.

Esta función se encarga de obtener las columnas de un DataFrame y clasificarlas
en numéricas y categóricas. Es una función común que puede ser utilizada en
diferentes etapas del pipeline, como en el preprocesamiento, entrenamiento
o producción.

V 0.1
"""

import pandas as pd


def get_columns(df: pd.DataFrame) -> tuple:
    """
    Obtiene las columnas del DataFrame y las categoriza.

    Numéricas: columnas que contienen datos numéricos (int, float).
    Categóricas: columnas que contienen datos categóricos (object, category).

    Parametros
    ----------
    df : pd.DataFrame
        El DataFrame del cual se obtendrán las columnas.

    Retorna
    ----------
    tuple
        Una tupla que contiene dos listas: una con las columnas numericas y
        otra con las columnas categoricas.

    Eleva
    ----------
    Exception
        Cualquier error inesperado al obtener las columnas.
    """
    try:
        columnas_categoricas = df.select_dtypes(include='object').columns
        columnas_numericas = df.select_dtypes(exclude='object').columns
        return columnas_numericas, columnas_categoricas
    except pd.errors.EmptyDataError:
        print("Error: El DataFrame está vacío.")
        return [], []
    except pd.errors.UnsupportedFunctionCall:
        print("Error: No se pueden aplicar funciones a este DataFrame.")
        return [], []
    except pd.errors.ParserError:
        print("Error: Problema al parsear el DataFrame.")
        return [], []
    except pd.errors.DtypeWarning:
        print("Advertencia: Problema con los tipos de datos en el DataFrame.")
        return [], []
