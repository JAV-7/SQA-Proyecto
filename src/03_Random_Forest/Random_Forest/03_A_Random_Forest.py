#!/usr/bin/env python
# coding: utf-8

# In[339]:


#####################################################
#
# Aplicar Random Forest a datos preprocesados con PCA
#
#####################################################
# Deben cargarse los archivos
# - T_train_final_objetivo.csv
# - T_test_final_objetivo.csv
# - pca_pipe_num.joblib
# - pca_metadata.json
# Devolverá
# expected_columns.json (columnas que deberán tener datos que nunca ha visto)
# feature_importance.csv (importancia de cada columna)
# modelo_random_forest.pkl (modelo ya entrenado)
# mi_random_forest_artifacts_bundle.zip
#####################################################

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import time
import os
import zipfile

##################################################################################################
Train = pd.read_csv("../../01_preprocessing_results/preprocessing/T_train_final_objetivo.csv")
Test = pd.read_csv("../../01_preprocessing_results/preprocessing/T_test_final_objetivo.csv")
##################################################################################################

X_train = Train.iloc[:, :-1]
y_train = Train.iloc[:, -1].to_numpy(dtype=float)

X_test = Test.iloc[:, :-1]
y_test = Test.iloc[:, -1].to_numpy(dtype=float)


# In[340]:


SEP = "___"

# Funciones auxiliares
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


# In[341]:


# 1) --- PRECOMPUTA CON TRAIN ---
blocks = build_nominal_blocks_by_prefix(X_train, SEP)
drop_cols = [cols[0] for cols in blocks.values() if len(cols) >= 2]  # primera de cada bloque


# In[342]:


# 2) --- PIPELINE CON RANDOM FOREST ---
arreglar_despeje = ColumnTransformer(
    transformers=[("drop_nominal_bases", "drop", drop_cols)],
    remainder="passthrough",
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
)

mi_random_forest = Pipeline([
    ("dropper", arreglar_despeje),
    ("rf", RandomForestRegressor(
        n_estimators=100,        # Número de árboles
        max_depth=20,          # Profundidad máxima (None = sin límite)
        min_samples_split=20,     # Mínimo de muestras para dividir un nodo
        min_samples_leaf=3,      # Mínimo de muestras en una hoja
        max_features='sqrt',     # Número de features a considerar en cada split
        random_state=42,         # Para reproducibilidad
        n_jobs=-1                # Usa todos los cores disponibles
    )),
])


# In[343]:


# 3) --- FIT & PRED ---
print("Training random forest")
mi_random_forest.fit(X_train, y_train)
print("Training completed")


# In[344]:


# Obtener importancia de features
feature_importances = mi_random_forest.named_steps["rf"].feature_importances_


# In[345]:


# Nombres de columnas después del dropper
feature_names = mi_random_forest.named_steps["dropper"].get_feature_names_out(X_train.columns)


# In[346]:


# Mostrar importancia de features
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importances
}).sort_values("importance", ascending=False)

print("\nFeatures importance (sorted):")
print(importance_df)


# In[347]:


# Parámetros del modelo
print("\nRandom Forest Parameters:")
print(f"Trees num: {mi_random_forest.named_steps['rf'].n_estimators}")
print(f"Max depth: {mi_random_forest.named_steps['rf'].max_depth}")
print(f"Max features: {mi_random_forest.named_steps['rf'].max_features}")


# In[348]:


# Evaluación en train y test
y_train_pred = mi_random_forest.predict(X_train)
y_test_pred = mi_random_forest.predict(X_test)

print("\n=== MÉTRICAS DE EVALUACIÓN ===")
print("\nTRAIN:")
print(f"  R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")

print("\nTEST:")
print(f"  R² Score: {r2_score(y_test, y_test_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")


# In[349]:


###### Guardado del modelo

# guarda el pipeline completo (dropper + RandomForestRegressor)
joblib.dump(mi_random_forest, "modelo_random_forest.pkl")


# In[350]:


# guarda el orden/esperado de columnas de entrenamiento
expected_cols = X_train.columns.tolist()
with open("expected_columns.json", "w", encoding="utf-8") as f:
    json.dump({"columns": expected_cols, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


# In[351]:


# guarda también la importancia de features
importance_df.to_csv("feature_importance.csv", index=False)

print("\nSaved artefacts:")
print("  - modelo_random_forest.pkl")
print("  - expected_columns.json")
print("  - feature_importance.csv")


# In[352]:


# Crear ZIP
dst_dir = r"mi_random_forest"
os.makedirs(dst_dir, exist_ok=True)
zip_path = os.path.join(dst_dir, "mi_random_forest_artifacts_bundle.zip")

# Archivos que quieres incluir
candidates = [
    "modelo_random_forest.pkl",
    "expected_columns.json",
    "feature_importance.csv",
]

present = [f for f in candidates if os.path.exists(f)]

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in present:
        zf.write(f, arcname=os.path.basename(f))  # guarda sin subcarpetas

print("\nZIP creado en:", zip_path)
print("Incluidos:", present)

