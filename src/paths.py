"""Define rutas relativas de scripts para entrenamiento y produccion."""

# TRAINING
DATA_CLEAN_PATH = "00_Data_Clean/00_A_Data_Clean.py"
PREPROCESSING_PATH = "01_Preprocessing/Preprocessing/01_A_Preprocessing.py"
LINEAL_REGRESSION_PATH = (
    "02_Lineal_Regression/Lineal_Regression/"
    "02_A_Lineal_Regression.py"
)
RANDOM_FOREST_PATH = "03_Random_Forest/Random_Forest/03_A_Random_Forest.py"
MODEL_EVALUATION_PATH = "04_Model_Evaluation/04_A_Model_Evaluation.py"
REGULARIZATION_PATH = "05_Regularization/Regularization/05_A_Regularization.py"

# PRODUCTION
PREPROCESSING_PROD_PATH = (
    "01_Preprocessing/Preprocessing_Production/"
    "01_B_Preprocessing_Production.py"
)
LINEAL_REGRESSION_PROD_PATH = (
    "02_Lineal_Regression/Lineal_Regression_Production/"
    "02_B_Lineal_Regression_Production.py"
)
RANDOM_FOREST_PROD_PATH = (
    "03_Random_Forest/Random_Forest_Production/"
    "03_B_Random_Forest_Production.py"
)
REGULARIZATION_PROD_PATH = (
    "05_Regularization/Regularization_Production/"
    "05_A_Regularization_Production.py"
)
