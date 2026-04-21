#!/usr/bin/env python
# coding: utf-8

# In[3]:


#####################################################
#
# APLICAR Regresión lineal a datos preprocesados con PCA
#
#####################################################
# Deben cargarse los archivos
# - T_train_final_objetivo.csv
# - T_test_final_objetivo.csv"
# - pca_pipe_num.joblib
# - pca_metadata.json
# Devolverá
# expected_columns.json (columnas que deberán tener datos que nunca ha visto)
# modelo_reg_lineal.pkl (modelo ya entrenado)
#####################################################

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

##################################################################################################
Train = pd.read_csv("../../01_preprocessing_results/preprocessing/T_train_final_objetivo.csv")
Test = pd.read_csv("../../01_preprocessing_results/preprocessing/T_test_final_objetivo.csv")
##################################################################################################

X_train = Train.iloc[:, :-1]
y_train = Train.iloc[:, -1].to_numpy(dtype=float)

X_test = Test.iloc[:, :-1]
y_test = Test.iloc[:, -1].to_numpy(dtype=float)

SEP = "___"  # Con esto encuentra las columnas Categoricas (One Hot)


# In[4]:


def is_binary_series(s: pd.Series):
    vals = pd.unique(s.dropna())
    return set(vals).issubset({0, 1}) or set(vals).issubset({0.0, 1.0})


def prefix_of(col: str, sep=SEP):
    return col.split(sep, 1)[0] if sep in col else None


def build_nominal_blocks_by_prefix(X: pd.DataFrame, sep=SEP):
    blocks = {}
    for c in X.columns:
        if sep in c and is_binary_series(X[c]):
            blocks.setdefault(prefix_of(c, sep), []).append(c)
    # respeta orden del CSV
    for k, v in blocks.items():
        blocks[k] = [c for c in X.columns if c in set(v)]
    return blocks


# In[5]:


# 1) --- PRECOMPUTA CON TRAIN ---
blocks = build_nominal_blocks_by_prefix(X_train, SEP)
drop_cols = [cols[0] for cols in blocks.values() if len(cols) >= 2]  # primera de cada bloque

# 2) --- PIPELINE SIN CLASES (usa ColumnTransformer para dropear fijo) ---
arreglar_despeje = ColumnTransformer(
    transformers=[("drop_nominal_bases", "drop", drop_cols)],
    remainder="passthrough",
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
)

mi_regresion_lineal = Pipeline([
    ("dropper", arreglar_despeje),
    ("linreg", LinearRegression(fit_intercept=True)),
])

# 3) --- FIT & PRED ---
mi_regresion_lineal.fit(X_train, y_train)


# In[6]:


# Intercepto y coeficientes del modelo dentro del pipeline
intercepto = mi_regresion_lineal.named_steps["linreg"].intercept_
coefs = mi_regresion_lineal.named_steps["linreg"].coef_

# Nombres de columnas después del dropper (lo más directo)
feature_names = mi_regresion_lineal.named_steps["dropper"].get_feature_names_out(X_train.columns)

# Mostrar los estimadores beta_0,beta_1,...,beta_p
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
print("Intercepto (beta0):", intercepto)
print(coef_df)


# In[7]:


###### Guardado del modelo

import joblib, json, time

# guarda el pipeline completo (dropper + LinearRegression)
joblib.dump(mi_regresion_lineal, "modelo_reg_lineal.pkl")

# guarda el orden/esperado de columnas de entrenamiento
expected_cols = X_train.columns.tolist()
with open("expected_columns.json", "w", encoding="utf-8") as f:
    json.dump({"columns": expected_cols, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)

print("Artefactos guardados:", "modelo_reg_lineal.pkl", "expected_columns.json")


# In[8]:


import os, zipfile, glob

# Carpeta destino en tu PC
dst_dir = r"mi_regresion_lineal"
os.makedirs(dst_dir, exist_ok=True)
zip_path = os.path.join(dst_dir, "mi_reg_lin_artifacts_bundle.zip")

# Archivos que quieres incluir (ajusta si te falta alguno)
candidates = [
    "modelo_reg_lineal.pkl",
    "expected_columns.json",
]

present = [f for f in candidates if os.path.exists(f)]
# Si quieres incluir una carpeta (p. ej., 'sample_data'), descomenta:
# for root, _, files in os.walk("sample_data"):
#     for f in files:
#         present.append(os.path.join(root, f))

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in present:
        zf.write(f, arcname=os.path.basename(f))  # guarda sin subcarpetas

print("ZIP creado en:", zip_path)
print("Incluidos:", present)


# In[9]:


# Evaluación en train y test
y_train_pred = mi_regresion_lineal.predict(X_train)
y_test_pred = mi_regresion_lineal.predict(X_test)

print("\n=== MÉTRICAS DE EVALUACIÓN ===")
print("\nTRAIN:")
print(f"  R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")

print("\nTEST:")
print(f"  R² Score: {r2_score(y_test, y_test_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")


# In[ ]:




