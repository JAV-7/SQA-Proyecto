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

from src.paths import (
    DATA_CLEAN_PATH,
    LINEAL_REGRESSION_PATH,
    RANDOM_FOREST_PATH,
    MODEL_EVALUATION_PATH,
    REGULARIZATION_PATH,
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

    stages_training = [
        ("Data Clean", DATA_CLEAN_PATH),
        ("Lineal Regression", LINEAL_REGRESSION_PATH),
        ("Random Forest", RANDOM_FOREST_PATH),
        ("Model Evaluation", MODEL_EVALUATION_PATH),
        ("Regularization", REGULARIZATION_PATH)
    ]

    stages_production = [
        ("Data Clean", DATA_CLEAN_PATH),
        ("Lineal Regression", LINEAL_REGRESSION_PROD_PATH),
        ("Random Forest", RANDOM_FOREST_PROD_PATH),
        ("Model Evaluation", MODEL_EVALUATION_PATH),
        ("Regularization", REGULARIZATION_PROD_PATH)
    ]

    for stage_name, stage_script in stages_training:
        print(f"\n=== Ejecutando etapa de entrenamiento: {stage_name} ===")
        result = subprocess.run(["python", stage_script], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error en etapa {stage_name}: {result.stderr}")
            return False

    for stage_name, stage_script in stages_production:
        print(f"\n=== Ejecutando etapa de producción: {stage_name} ===")
        result = subprocess.run(["python", stage_script], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error en etapa {stage_name}: {result.stderr}")
            return False

    print("\n=== Todas las etapas completadas exitosamente ===")
    return True 

if __name__ == "__main__":
    main()