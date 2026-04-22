"""
main.py

Archivo orquestrador para ejecutar cada etapa del pipeline.

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa

V 0.0

"""

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

from paths import (
    DATA_CLEAN_PATH,
    PREPROCESSING_PATH,
    LINEAL_REGRESSION_PATH,
    RANDOM_FOREST_PATH,
    MODEL_EVALUATION_PATH,
    REGULARIZATION_PATH,
    PREPROCESSING_PROD_PATH,
    LINEAL_REGRESSION_PROD_PATH,
    RANDOM_FOREST_PROD_PATH,
    REGULARIZATION_PROD_PATH
)

def welcome_message():
    print("Bienvenidos al pipeline de ML para el proyecto de Calidad de Software")
    print("Ejecutando cada etapa en orden: Data Clean -> Lineal Regression -> Random Forest -> Regularization")
    print("Cada etapa imprimirá su progreso y resultados. Al finalizar, se mostrarán las predicciones finales en producción.")

def main() -> bool:
    """Ejecuta cada etapa del pipeline en orden"""
    welcome_message()
    python_exe = sys.executable # Asegurar que se ejecute con el mismo intérprete de Python
    src_dir = Path(__file__).resolve().parent

    stages = [
        ("Data Clean", DATA_CLEAN_PATH),
        ("Preprocessing", PREPROCESSING_PATH),
        ("Preprocessing Production", PREPROCESSING_PROD_PATH),
        ("Lineal Regression", LINEAL_REGRESSION_PATH),
        ("Lineal Regression Production", LINEAL_REGRESSION_PROD_PATH),
        ("Random Forest", RANDOM_FOREST_PATH),
        ("Random Forest", RANDOM_FOREST_PROD_PATH),
        ("Model Evaluation", MODEL_EVALUATION_PATH),
        ("Regularization", REGULARIZATION_PATH),
        ("Regularization Production", REGULARIZATION_PROD_PATH)

    ]


    stage_iterable = tqdm(stages, total=len(stages), desc="Pipeline", unit="etapa")

    for stage_name, stage_script in stage_iterable:
        stage_path = src_dir / stage_script

        if tqdm and hasattr(stage_iterable, "set_postfix_str"):
            stage_iterable.set_postfix_str(stage_name)

        print(f"\n=== Ejecutando etapa de: {stage_name} ===")
        result = subprocess.run(
            [python_exe, str(stage_path)],
            capture_output=True,
            text=True,
            cwd=str(src_dir),
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            if tqdm and hasattr(stage_iterable, "close"):
                stage_iterable.close()
            print(f"Error en etapa {stage_name}: {result.stderr}")
            return False

    print("\n=== Todas las etapas completadas exitosamente ===")
    return True 

if __name__ == "__main__":
    main()