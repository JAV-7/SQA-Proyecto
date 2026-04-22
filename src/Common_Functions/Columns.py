import pandas as pd


def get_columns(df : pd.DataFrame) -> tuple:
    """
    Obtiene las columnas del DataFrame y las categoriza en numericas
    y categoricas.

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
        columnas_categoricas = df.select_dtypes(include ='object').columns
        columnas_numericas = df.select_dtypes(exclude ='object').columns
        return columnas_numericas, columnas_categoricas
    except Exception as e:
        print(f"Error al obtener las columnas: {e}")
        return [], []