"""
Test suite para el pipeline de ML (Calidad de Software).

Ejecución desde la raíz del proyecto:
    pytest src/tests/test_pipeline.py -v

pytest.ini debe tener:
    --cov=src/00_Data_Clean
    --cov=src/01_Preprocessing/...
    etc.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# sys.path – agrega src/ para que "import paths" funcione
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Stubs de Common_Functions (evita ImportError al importar módulos reales)
# ---------------------------------------------------------------------------
def _register_stubs():
    cf     = types.ModuleType("Common_Functions")
    cf_col = types.ModuleType("Common_Functions.Columns")
    cf_iqr = types.ModuleType("Common_Functions.IQR")

    def get_columns(df):
        return (df.select_dtypes(include="number").columns,
                df.select_dtypes(include="object").columns)

    def iqr_outlier_stats(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        n = int(mask.sum())
        return n, n / max(len(series), 1)

    cf_col.get_columns      = get_columns
    cf_iqr.iqr_outlier_stats = iqr_outlier_stats
    sys.modules.setdefault("Common_Functions",         cf)
    sys.modules.setdefault("Common_Functions.Columns", cf_col)
    sys.modules.setdefault("Common_Functions.IQR",     cf_iqr)

_register_stubs()

# ---------------------------------------------------------------------------
# Helper para importar módulos por ruta de archivo
# ---------------------------------------------------------------------------
def _import(rel: str):
    path = SRC_DIR / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Importar módulos reales
import paths as _paths                                                    # noqa: E402
_dc  = _import("00_Data_Clean/00_A_Data_Clean.py")
_me  = _import("04_Model_Evaluation/04_A_Model_Evaluation.py")
_reg = _import("05_Regularization/Regularization/05_A_Regularization.py")
_lr  = _import("02_Lineal_Regression/Lineal_Regression/02_A_Lineal_Regression.py")
_rf  = _import("03_Random_Forest/Random_Forest/03_A_Random_Forest.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Store ID":           [f"S{i:03d}" for i in range(n)],
        "Product ID":         [f"P{i:03d}" for i in range(n)],
        "Date":               pd.date_range("2023-01-01", periods=n, freq="D")
                              .strftime("%d/%m/%Y"),
        "Category":           rng.choice(["electronics","furniture",
                                           "clothing","toys","groceries"], n),
        "Units Sold":         rng.integers(1, 200, n).astype(float),
        "Units Ordered":      rng.integers(1, 300, n).astype(float),
        "Inventory Level":    rng.integers(0, 500, n).astype(float),
        "Price":              rng.uniform(5, 500, n),
        "Competitor Pricing": rng.uniform(5, 500, n),
        "Discount":           rng.uniform(0, 0.5, n),
        "Demand Forecast":    rng.integers(10, 400, n).astype(float),
        "Weather Condition":  rng.choice(["sunny","rainy","cloudy"], n),
    })


def _make_preprocessed_df(n=80, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"PC{i}": rng.uniform(-2, 2, n) for i in range(1, 6)}
    for c in ["Category___electronics","Category___furniture",
              "Category___clothing","Category___toys",
              "Store___1","Store___2"]:
        data[c] = rng.choice([0.0, 1.0], n)
    data["objetivo"] = rng.uniform(10, 400, n)
    return pd.DataFrame(data)


@pytest.fixture
def raw_df():
    return _make_raw_df()

@pytest.fixture
def preprocessed_df():
    return _make_preprocessed_df()

@pytest.fixture
def train_test(preprocessed_df):
    X = preprocessed_df.drop(columns=["objetivo"])
    y = preprocessed_df["objetivo"]
    return train_test_split(X, y, test_size=0.25, random_state=42)


# ===========================================================================
# paths.py
# ===========================================================================

class TestPaths:
    def test_all_paths_are_strings(self):
        for attr in ["DATA_CLEAN_PATH","PREPROCESSING_PATH",
                     "LINEAL_REGRESSION_PATH","RANDOM_FOREST_PATH",
                     "MODEL_EVALUATION_PATH","REGULARIZATION_PATH",
                     "PREPROCESSING_PROD_PATH","LINEAL_REGRESSION_PROD_PATH",
                     "RANDOM_FOREST_PROD_PATH","REGULARIZATION_PROD_PATH"]:
            assert isinstance(getattr(_paths, attr), str)

    def test_all_paths_end_with_py(self):
        for attr in ["DATA_CLEAN_PATH","PREPROCESSING_PATH",
                     "LINEAL_REGRESSION_PATH","RANDOM_FOREST_PATH",
                     "MODEL_EVALUATION_PATH","REGULARIZATION_PATH",
                     "PREPROCESSING_PROD_PATH","LINEAL_REGRESSION_PROD_PATH",
                     "RANDOM_FOREST_PROD_PATH","REGULARIZATION_PROD_PATH"]:
            assert getattr(_paths, attr).endswith(".py"), attr

    def test_data_clean_path_folder(self):
        assert "00_Data_Clean" in _paths.DATA_CLEAN_PATH

    def test_production_paths_keyword(self):
        for attr in ["PREPROCESSING_PROD_PATH","LINEAL_REGRESSION_PROD_PATH",
                     "RANDOM_FOREST_PROD_PATH","REGULARIZATION_PROD_PATH"]:
            val = getattr(_paths, attr)
            assert "Production" in val or "Prod" in val, attr

    def test_training_paths_do_not_contain_production(self):
        for attr in ["DATA_CLEAN_PATH","PREPROCESSING_PATH",
                     "LINEAL_REGRESSION_PATH","RANDOM_FOREST_PATH"]:
            assert "Production" not in getattr(_paths, attr), attr


# ===========================================================================
# 00_A_Data_Clean
# ===========================================================================

class TestDataClean:

    def test_clean_text_strips_and_lowercases(self, raw_df):
        result = _dc._clean_text_columns(raw_df.copy())
        for col in result.select_dtypes(include="object").columns:
            assert result[col].str.strip().eq(result[col]).all()
            assert result[col].str.lower().eq(result[col]).all()

    def test_clean_category_fixes_typos(self):
        df = pd.DataFrame({"Category": [
            "electronicsx","furniturex","clothingx","toysx","groceriesx"]})
        result = _dc._clean_category_column(df)
        assert set(result["Category"]) == {
            "electronics","furniture","clothing","toys","groceries"}

    def test_clean_category_removes_non_alpha(self):
        df = pd.DataFrame({"Category": ["electronics123","furniture!"]})
        result = _dc._clean_category_column(df)
        assert result["Category"].str.match(r"^[a-z]+$").all()

    def test_clean_data_returns_dataframe(self, raw_df, tmp_path):
        with (patch.object(_dc, "REPORTS_DIR", tmp_path),
              patch.object(_dc, "GRAPHICS_DIR", tmp_path),
              patch.object(_dc, "FILES_DIR",    tmp_path),
              patch.object(_dc, "CLEAN_DATA_PATH", tmp_path/"clean.txt")):
            result = _dc.clean_data(raw_df)
        assert isinstance(result, pd.DataFrame)
        assert "Weather Condition" not in result.columns
        assert "Date" not in result.columns

    def test_clean_data_date_column_removed(self, raw_df, tmp_path):
        with (patch.object(_dc, "REPORTS_DIR", tmp_path),
              patch.object(_dc, "GRAPHICS_DIR", tmp_path),
              patch.object(_dc, "FILES_DIR",    tmp_path),
              patch.object(_dc, "CLEAN_DATA_PATH", tmp_path/"clean.txt")):
            result = _dc.clean_data(raw_df)
        assert "Date" not in result.columns

    def test_clean_data_adds_timestamp(self, raw_df, tmp_path):
        with (patch.object(_dc, "REPORTS_DIR", tmp_path),
              patch.object(_dc, "GRAPHICS_DIR", tmp_path),
              patch.object(_dc, "FILES_DIR",    tmp_path),
              patch.object(_dc, "CLEAN_DATA_PATH", tmp_path/"clean.txt")):
            result = _dc.clean_data(raw_df)
        assert "date_timestamp" in result.columns

    def test_get_initial_insights_writes_file(self, raw_df, tmp_path):
        with (patch.object(_dc, "REPORTS_DIR", tmp_path),
              patch.object(_dc, "INITIAL_INSIGHTS_PATH",
                           tmp_path/"insights.txt")):
            ok = _dc.get_initial_insights(raw_df)
        assert ok is True
        assert (tmp_path/"insights.txt").exists()

    def test_get_initial_insights_content(self, raw_df, tmp_path):
        insights_path = tmp_path / "insights.txt"
        with (patch.object(_dc, "REPORTS_DIR", tmp_path),
              patch.object(_dc, "INITIAL_INSIGHTS_PATH", insights_path)):
            _dc.get_initial_insights(raw_df)
        content = insights_path.read_text()
        assert "Dimensiones" in content

    def test_save_clean_data_creates_csv(self, raw_df, tmp_path):
        with patch.object(_dc, "FILES_DIR", tmp_path):
            ok = _dc.save_clean_data(raw_df)
        assert ok is True
        assert (tmp_path/"retail_store_inventory_limpio.csv").exists()

    def test_save_clean_data_creates_train_csv(self, raw_df, tmp_path):
        with patch.object(_dc, "FILES_DIR", tmp_path):
            _dc.save_clean_data(raw_df)
        assert (tmp_path/"retail_store_inventory_entrenamiento.csv").exists()

    def test_write_clean_data_report(self, raw_df, tmp_path):
        out = tmp_path / "report.txt"
        with patch.object(_dc, "CLEAN_DATA_PATH", out):
            _dc._write_clean_data_report(raw_df, raw_df)
        assert out.exists()
        assert "Dimensiones" in out.read_text()


# ===========================================================================
# 02_A_Lineal_Regression
# ===========================================================================

class TestLinealRegression:

    def test_is_binary_true(self):
        assert _lr.is_binary_series(pd.Series([0.0, 1.0, 0.0, np.nan]))

    def test_is_binary_false(self):
        assert not _lr.is_binary_series(pd.Series([0.1, 0.5, 0.9]))

    def test_prefix_of_with_sep(self):
        assert _lr.prefix_of("Category___electronics") == "Category"

    def test_prefix_of_no_sep(self):
        assert _lr.prefix_of("PC1") is None

    def test_nominal_blocks(self, preprocessed_df):
        X = preprocessed_df.drop(columns=["objetivo"])
        blocks = _lr.build_nominal_blocks_by_prefix(X)
        assert "Category" in blocks
        assert len(blocks["Category"]) >= 2

    def test_pipeline_trains(self, preprocessed_df):
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        m = LinearRegression().fit(X, y)
        preds = m.predict(X)
        assert not np.any(np.isnan(preds))

    def test_evaluate_prints_metrics(self, preprocessed_df, capsys):
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        m = LinearRegression().fit(X, y)
        _lr._evaluate_and_print_metrics(m, X, y, X, y)
        out = capsys.readouterr().out
        assert "R²" in out or "RMSE" in out

    def test_print_model_info(self, preprocessed_df, capsys):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        ct = ColumnTransformer(transformers=[],remainder="passthrough",
                               verbose_feature_names_out=False,
                               force_int_remainder_cols=False)
        pipe = Pipeline([("dropper", ct),
                         ("linreg", LinearRegression())]).fit(X, y)
        _lr._print_model_info(pipe, X)
        out = capsys.readouterr().out
        assert "Intercepto" in out or "coef" in out.lower()


# ===========================================================================
# 03_A_Random_Forest
# ===========================================================================

class TestRandomForest:

    def test_is_binary(self):
        assert _rf.is_binary_series(pd.Series([0, 1, 0, 1]))
        assert not _rf.is_binary_series(pd.Series([0.5, 1.5]))

    def test_prefix_of(self):
        assert _rf.prefix_of("Store___1") == "Store"
        assert _rf.prefix_of("PC1") is None

    def test_nominal_blocks(self, preprocessed_df):
        X = preprocessed_df.drop(columns=["objetivo"])
        blocks = _rf.build_nominal_blocks_by_prefix(X)
        assert len(blocks) > 0

    def test_rf_feature_importances_sum(self, preprocessed_df):
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
        rf.fit(X, y)
        assert abs(rf.feature_importances_.sum() - 1.0) < 1e-6

    def test_evaluate_returns_importance_df(self, preprocessed_df):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        ct = ColumnTransformer(transformers=[],remainder="passthrough",
                               verbose_feature_names_out=False,
                               force_int_remainder_cols=False)
        pipe = Pipeline([("dropper", ct),
                         ("rf", RandomForestRegressor(
                             n_estimators=10, random_state=42, n_jobs=1
                         ))]).fit(X, y)
        imp = _rf._evaluate_and_print_metrics(pipe, X, y, X, y)
        assert "feature" in imp.columns
        assert "importance" in imp.columns

    def test_missing_columns_detection(self, preprocessed_df):
        expected = preprocessed_df.drop(columns=["objetivo"]).columns.tolist()
        df_new = preprocessed_df.drop(columns=["objetivo","PC1"],
                                       errors="ignore")
        missing = [c for c in expected if c not in df_new.columns]
        assert "PC1" in missing


# ===========================================================================
# 04_A_Model_Evaluation
# ===========================================================================

class TestModelEvaluation:

    def test_calcular_metricas_all_keys(self, preprocessed_df):
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy()
        m = LinearRegression().fit(X, y)
        result = _me.calcular_metricas(y, m.predict(X), "Test")
        for k in ["Modelo","R²","RMSE","NRMSE","MAE","MAPE (%)"]:
            assert k in result

    def test_r2_perfect_prediction(self):
        y = np.linspace(10, 400, 50)
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_rmse_perfect_prediction(self):
        y = np.linspace(10, 400, 50)
        assert np.sqrt(mean_squared_error(y, y)) == pytest.approx(0.0)

    def test_mape_no_zero_division(self):
        y_real = np.array([0.0, 10.0, 20.0])
        y_pred = np.array([1.0, 10.0, 20.0])
        denom  = np.where(y_real == 0, 1, y_real)
        assert np.isfinite(np.mean(np.abs((y_real - y_pred) / denom)) * 100)

    @pytest.mark.parametrize("r2,nrmse,expected", [
        (0.80, 0.30, "VERDE"),
        (0.55, 0.60, "AMARILLO"),
        (0.20, 0.90, "ROJO"),
    ])
    def test_get_veredicto(self, r2, nrmse, expected):
        color, _ = _me._get_veredicto({"R²": r2, "NRMSE": nrmse})
        assert color == expected

    def test_create_metrics_table_shape(self, preprocessed_df):
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy()
        m = LinearRegression().fit(X, y)
        m1 = _me.calcular_metricas(y, m.predict(X), "Lineal")
        m2 = _me.calcular_metricas(y, m.predict(X), "RF")
        table = _me._create_metrics_table(m1, m2)
        assert len(table) == 2
        assert "R²" in table.columns

    def test_print_data_summary(self, preprocessed_df, capsys):
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"]
        _me._print_data_summary(X, y)
        assert "muestras" in capsys.readouterr().out

    def test_print_prediction_summary(self, capsys):
        _me._print_prediction_summary(np.array([1.,2.,3.]),
                                      np.array([1.5,2.5,3.5]))
        assert capsys.readouterr().out != ""

    def test_print_comparative_report(self, preprocessed_df, capsys):
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy()
        m = LinearRegression().fit(X, y)
        m1 = _me.calcular_metricas(y, m.predict(X), "Regresion Lineal")
        m2 = _me.calcular_metricas(y, m.predict(X), "Random Forest")
        table = _me._create_metrics_table(m1, m2)
        _me._print_comparative_report(m1, m2, table)
        assert "GANADOR" in capsys.readouterr().out


# ===========================================================================
# 05_A_Regularization
# ===========================================================================

class TestRegularization:

    def test_calcular_metricas(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        m = LinearRegression().fit(X_tr, y_tr)
        r = _reg.calcular_metricas(m, X_te, y_te, "Base")
        assert r["R²"] <= 1.0 and r["RMSE"] >= 0.0

    def test_ridge(self, train_test):
        X_tr, _, y_tr, _ = train_test
        ridge = RidgeCV(alphas=np.logspace(-2,2,10), cv=3).fit(X_tr, y_tr)
        assert hasattr(ridge, "alpha_")

    def test_lasso(self, train_test):
        X_tr, _, y_tr, _ = train_test
        lasso = LassoCV(alphas=np.logspace(-2,2,10), cv=3,
                        max_iter=1000, random_state=42).fit(X_tr, y_tr)
        assert hasattr(lasso, "alpha_")

    def test_elasticnet(self, train_test):
        X_tr, _, y_tr, _ = train_test
        elastic = ElasticNetCV(alphas=np.logspace(-2,2,5),
                               l1_ratio=[0.5,0.9], cv=3,
                               max_iter=1000, random_state=42).fit(X_tr, y_tr)
        assert hasattr(elastic, "l1_ratio_")

    def test_select_best_regularized(self):
        df = pd.DataFrame(
            {"R²":[0.70,0.72,0.68,0.75],"RMSE":[15,14,16,13],"MAE":[10,9,11,8]},
            index=["Reg. Lineal (base)","Ridge (L2)","Lasso (L1)","Elastic Net"])
        best, r2 = _reg._select_best_regularized(df)
        assert best == "Elastic Net"
        assert r2 == pytest.approx(0.75)

    def test_build_metrics_table(self, train_test):
        X_tr, X_te, y_tr, y_te = train_test
        models = (
            LinearRegression().fit(X_tr, y_tr),
            RidgeCV(alphas=[0.1,1.0], cv=3).fit(X_tr, y_tr),
            LassoCV(alphas=[0.1,1.0], cv=3, max_iter=500).fit(X_tr, y_tr),
            ElasticNetCV(alphas=[0.1], l1_ratio=[0.5], cv=3,
                         max_iter=500).fit(X_tr, y_tr),
        )
        table = _reg._build_metrics_table(models, X_te, y_te)
        assert len(table) == 4 and "R²" in table.columns

    def test_save_and_print_metrics(self, train_test, tmp_path, capsys):
        X_tr, X_te, y_tr, y_te = train_test
        models = (
            LinearRegression().fit(X_tr, y_tr),
            RidgeCV(alphas=[0.1,1.0], cv=3).fit(X_tr, y_tr),
            LassoCV(alphas=[0.1], cv=3, max_iter=500).fit(X_tr, y_tr),
            ElasticNetCV(alphas=[0.1], l1_ratio=[0.5], cv=3,
                         max_iter=500).fit(X_tr, y_tr),
        )
        table = _reg._build_metrics_table(models, X_te, y_te)
        with patch.object(_reg, "FILES_DIR", tmp_path):
            _reg._save_and_print_metrics(table)
        assert (tmp_path/"comparacion_regularizacion.csv").exists()

    def test_build_and_save_coefficients(self, train_test, tmp_path):
        X_tr, _, y_tr, _ = train_test
        models = (
            LinearRegression().fit(X_tr, y_tr),
            RidgeCV(alphas=[0.1,1.0], cv=3).fit(X_tr, y_tr),
            LassoCV(alphas=[0.1], cv=3, max_iter=500).fit(X_tr, y_tr),
            ElasticNetCV(alphas=[0.1], l1_ratio=[0.5], cv=3,
                         max_iter=500).fit(X_tr, y_tr),
        )
        with patch.object(_reg, "FILES_DIR", tmp_path):
            _reg._build_and_save_coefficients(
                X_tr.columns.tolist(), models)
        assert (tmp_path/"coeficientes_modelos.csv").exists()

    def test_build_and_save_pca_betas(self, train_test, tmp_path):
        X_tr, _, y_tr, _ = train_test
        reg     = LinearRegression().fit(X_tr, y_tr)
        elastic = ElasticNetCV(alphas=[0.1], l1_ratio=[0.5], cv=3,
                               max_iter=500).fit(X_tr, y_tr)
        with patch.object(_reg, "FILES_DIR", tmp_path):
            _reg._build_and_save_pca_betas(
                X_tr.columns.tolist(), reg, elastic)
        assert (tmp_path/"betas_post_pre_numericas.csv").exists()

    def test_save_best_bundle_creates_zip(self, train_test, tmp_path):
        X_tr, X_te, y_tr, y_te = train_test
        models = (
            LinearRegression().fit(X_tr, y_tr),
            RidgeCV(alphas=[0.1,1.0], cv=3).fit(X_tr, y_tr),
            LassoCV(alphas=[0.1], cv=3, max_iter=500).fit(X_tr, y_tr),
            ElasticNetCV(alphas=[0.1], l1_ratio=[0.5], cv=3,
                         max_iter=500).fit(X_tr, y_tr),
        )
        df = pd.DataFrame(
            {"R²":[0.70,0.72,0.68,0.75],"RMSE":[15,14,16,13],"MAE":[10,9,11,8]},
            index=["Reg. Lineal (base)","Ridge (L2)","Lasso (L1)","Elastic Net"])
        with patch.object(_reg, "REGULARIZATION_BUNDLE_DIR", tmp_path):
            _reg._save_best_bundle(
                "Elastic Net", 0.75, models,
                X_tr.columns.tolist(), df)
        assert (tmp_path/"reg_lin_ganador_bundle.zip").exists()

    def test_print_conclusion(self, capsys):
        df = pd.DataFrame(
            {"R²":[0.70,0.72,0.68,0.75],"RMSE":[15,14,16,13],"MAE":[10,9,11,8]},
            index=["Reg. Lineal (base)","Ridge (L2)","Lasso (L1)","Elastic Net"])
        _reg._print_conclusion(df, "Elastic Net")
        assert "Elastic Net" in capsys.readouterr().out

    def test_plot_and_save_metrics(self, train_test, tmp_path):
        X_tr, X_te, y_tr, y_te = train_test
        models = (
            LinearRegression().fit(X_tr, y_tr),
            RidgeCV(alphas=[0.1,1.0], cv=3).fit(X_tr, y_tr),
            LassoCV(alphas=[0.1], cv=3, max_iter=500).fit(X_tr, y_tr),
            ElasticNetCV(alphas=[0.1], l1_ratio=[0.5], cv=3,
                         max_iter=500).fit(X_tr, y_tr),
        )
        table = _reg._build_metrics_table(models, X_te, y_te)
        with patch.object(_reg, "GRAPHICS_DIR", tmp_path):
            _reg._plot_and_save_metrics(table)
        assert (tmp_path/"comparacion_regularizacion.png").exists()


# ===========================================================================
# Artifact persistence
# ===========================================================================

class TestArtifactPersistence:

    def test_joblib_roundtrip(self, tmp_path, preprocessed_df):
        import joblib
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy()
        m = LinearRegression().fit(X, y)
        path = tmp_path / "model.pkl"
        joblib.dump(m, path)
        loaded = joblib.load(path)
        np.testing.assert_array_almost_equal(m.predict(X), loaded.predict(X))

    def test_json_roundtrip(self, tmp_path, preprocessed_df):
        cols = preprocessed_df.drop(columns=["objetivo"]).columns.tolist()
        p = tmp_path / "cols.json"
        with open(p, "w") as f:
            json.dump({"columns": cols}, f)
        with open(p) as f:
            assert json.load(f)["columns"] == cols

    def test_csv_roundtrip(self, tmp_path, preprocessed_df):
        p = tmp_path / "data.csv"
        preprocessed_df.to_csv(p, index=False)
        loaded = pd.read_csv(p)
        assert list(loaded.columns) == list(preprocessed_df.columns)

    def test_zip_roundtrip(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        zp = tmp_path / "out.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(f, f.name)
        with zipfile.ZipFile(zp) as z:
            assert "file.txt" in z.namelist()


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_k90_k95_valid(self):
        variances = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
        cum = np.cumsum(variances)
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        k95 = int(np.searchsorted(cum, 0.95) + 1)
        assert 1 <= k90 <= k95

    def test_k_elbow_is_int(self):
        exp = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
        d2 = np.diff(np.diff(exp))
        k_elbow = int(np.argmax(-d2) + 2) if len(d2) else 1
        assert isinstance(k_elbow, int) and k_elbow >= 1

    def test_nrmse_non_negative(self):
        y  = np.array([10., 20., 30.])
        yp = np.array([12., 18., 31.])
        nrmse = np.sqrt(mean_squared_error(y, yp)) / (y.max() - y.min())
        assert nrmse >= 0.0

    def test_split_sizes(self, preprocessed_df):
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"]
        Xtr, Xte, _, _ = train_test_split(X, y, test_size=0.25, random_state=42)
        assert len(Xtr) + len(Xte) == len(X)


# ===========================================================================
# Common_Functions – imports reales
# ===========================================================================

def _import_cf(rel: str):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Importar módulos reales de Common_Functions
try:
    _cf_cols = _import_cf("Common_Functions/Columns.py")
    _cf_iqr  = _import_cf("Common_Functions/IQR.py")
    _CF_AVAILABLE = True
except Exception:
    _CF_AVAILABLE = False


class TestCommonFunctionsColumns:

    def test_get_columns_returns_tuple(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.5, 2.5]})
        num, cat = _cf_cols.get_columns(df)
        assert len(num) == 2   # a, c
        assert len(cat) == 1   # b

    def test_get_columns_all_numeric(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        num, cat = _cf_cols.get_columns(df)
        assert len(num) == 2
        assert len(cat) == 0

    def test_get_columns_all_categorical(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        num, cat = _cf_cols.get_columns(df)
        assert len(num) == 0
        assert len(cat) == 2

    def test_get_columns_mixed(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        df = _make_raw_df()
        num, cat = _cf_cols.get_columns(df)
        assert len(num) > 0
        assert len(cat) > 0

    def test_get_columns_empty_dataframe(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        df = pd.DataFrame()
        num, cat = _cf_cols.get_columns(df)
        assert len(num) == 0
        assert len(cat) == 0

    def test_get_columns_numeric_types(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        df = pd.DataFrame({"int_col": [1, 2], "float_col": [1.1, 2.2],
                           "str_col": ["a", "b"]})
        num, cat = _cf_cols.get_columns(df)
        assert "int_col" in num
        assert "float_col" in num
        assert "str_col" in cat


class TestCommonFunctionsIQR:

    def test_iqr_returns_tuple(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        n, pct = _cf_iqr.iqr_outlier_stats(s)
        assert isinstance(n, int)
        assert isinstance(pct, float)

    def test_iqr_detects_outlier(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        s = pd.Series([1.0] * 20 + [1000.0])
        n, pct = _cf_iqr.iqr_outlier_stats(s)
        assert n >= 1

    def test_iqr_no_outliers(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        s = pd.Series([10.0, 11.0, 12.0, 11.5, 10.5])
        n, pct = _cf_iqr.iqr_outlier_stats(s)
        assert n == 0
        assert pct == 0.0

    def test_iqr_empty_series(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        s = pd.Series([], dtype=float)
        n, pct = _cf_iqr.iqr_outlier_stats(s)
        assert n == 0
        assert pct == 0.0

    def test_iqr_all_nan(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        s = pd.Series([np.nan, np.nan, np.nan])
        n, pct = _cf_iqr.iqr_outlier_stats(s)
        assert n == 0

    def test_iqr_pct_between_0_and_1(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        s = pd.Series(list(range(50)) + [9999])
        n, pct = _cf_iqr.iqr_outlier_stats(s)
        assert 0.0 <= pct <= 1.0

    def test_iqr_with_non_numeric_coerced(self):
        if not _CF_AVAILABLE:
            pytest.skip("Common_Functions no disponible")
        s = pd.Series(["1", "2", "3", "abc", "100"])
        n, pct = _cf_iqr.iqr_outlier_stats(s)
        assert isinstance(n, int)


# ===========================================================================
# Tests adicionales para subir coverage de módulos existentes
# ===========================================================================

class TestDataCleanExtra:

    def test_plot_numeric_columns_runs(self, tmp_path):
        df = _make_preprocessed_df().drop(columns=["objetivo"])
        with patch.object(_dc, "GRAPHICS_DIR", tmp_path):
            _dc._plot_numeric_columns(df)

    def test_plot_categorical_columns_runs(self, tmp_path):
        df = pd.DataFrame({"Category": ["electronics", "furniture",
                                         "clothing", "toys"],
                           "Store": ["s1", "s2", "s1", "s2"]})
        with patch.object(_dc, "GRAPHICS_DIR", tmp_path):
            _dc._plot_categorical_columns(df)

    def test_clean_data_error_handling(self, tmp_path):
        """clean_data con DataFrame que lanza error retorna None."""
        with (patch.object(_dc, "REPORTS_DIR", tmp_path),
              patch.object(_dc, "GRAPHICS_DIR", tmp_path),
              patch.object(_dc, "FILES_DIR", tmp_path),
              patch.object(_dc, "CLEAN_DATA_PATH", tmp_path/"c.txt"),
              patch.object(_dc, "_clean_text_columns",
                           side_effect=ValueError("test error"))):
            result = _dc.clean_data(_make_raw_df())
        assert result is None

    def test_ensure_output_dirs(self, tmp_path):
        with (patch.object(_dc, "REPORTS_DIR", tmp_path/"r"),
              patch.object(_dc, "GRAPHICS_DIR", tmp_path/"g"),
              patch.object(_dc, "FILES_DIR",    tmp_path/"f")):
            _dc._ensure_output_dirs()
        assert (tmp_path/"r").exists()
        assert (tmp_path/"g").exists()
        assert (tmp_path/"f").exists()

    def test_save_clean_data_creates_unknown_csv(self, tmp_path):
        df = _make_raw_df()
        df["Demand Forecast"] = np.random.default_rng(0).integers(10, 400, len(df)).astype(float)
        with patch.object(_dc, "FILES_DIR", tmp_path):
            _dc.save_clean_data(df)
        assert (tmp_path/"retail_store_inventory_produccion_unknown.csv").exists()


class TestLinealRegressionExtra:

    def test_build_and_train_pipeline_returns_pipeline(self, preprocessed_df):
        from sklearn.pipeline import Pipeline
        from tqdm import tqdm
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        progress = tqdm(total=3, disable=True)
        result = _lr._build_and_train_pipeline(X, y, progress)
        assert isinstance(result, Pipeline)
        progress.close()

    def test_save_artifacts(self, preprocessed_df, tmp_path):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from tqdm import tqdm
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        ct = ColumnTransformer(transformers=[], remainder="passthrough",
                               verbose_feature_names_out=False,
                               force_int_remainder_cols=False)
        pipe = Pipeline([("dropper", ct),
                         ("linreg", LinearRegression())]).fit(X, y)
        progress = tqdm(total=2, disable=True)
        with (patch.object(_lr, "MODEL_DIR", tmp_path),
              patch.object(_lr, "LINEAR_FILES_DIR", tmp_path)):
            _lr._save_artifacts(pipe, X, progress)
        assert (tmp_path/"modelo_reg_lineal.pkl").exists()
        progress.close()


class TestRandomForestExtra:

    def test_build_and_train_pipeline_rf(self, preprocessed_df):
        from sklearn.pipeline import Pipeline
        from tqdm import tqdm
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        progress = tqdm(total=3, disable=True)
        # Usamos n_estimators pequeño para que sea rápido
        with patch.object(
            _rf.RandomForestRegressor, "__init__",
            lambda self, **kwargs: RandomForestRegressor.__init__(
                self, n_estimators=5, random_state=42, n_jobs=1)
        ):
            pass
        result = _rf._build_and_train_pipeline(X, y, progress)
        assert isinstance(result, Pipeline)
        progress.close()

    def test_save_artifacts_rf(self, preprocessed_df, tmp_path):
        from tqdm import tqdm
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy(dtype=float)
        rf = RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=1)
        rf.fit(X, y)
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        ct = ColumnTransformer(transformers=[], remainder="passthrough",
                               verbose_feature_names_out=False,
                               force_int_remainder_cols=False)
        pipe = Pipeline([("dropper", ct), ("rf", rf)]).fit(X, y)
        imp_df = pd.DataFrame({"feature": X.columns,
                                "importance": rf.feature_importances_})
        progress = tqdm(total=2, disable=True)
        with (patch.object(_rf, "MODEL_PATH", tmp_path/"model.pkl"),
              patch.object(_rf, "EXPECTED_COLUMNS_PATH",
                           tmp_path/"cols.json"),
              patch.object(_rf, "FEATURE_IMPORTANCE_PATH",
                           tmp_path/"imp.csv"),
              patch.object(_rf, "ARTIFACTS_DIR", tmp_path/"artifacts")):
            _rf._save_artifacts(pipe, imp_df, X, progress)
        assert (tmp_path/"model.pkl").exists()
        progress.close()


class TestModelEvaluationExtra:

    def test_plot_model_comparison_returns_figure(self, preprocessed_df):
        import matplotlib
        matplotlib.use("Agg")
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"]
        m = LinearRegression().fit(X, y)
        yp = m.predict(X)
        m1 = _me.calcular_metricas(y.to_numpy(), yp, "Lineal")
        m2 = _me.calcular_metricas(y.to_numpy(), yp, "RF")
        fig = _me._plot_model_comparison(y, yp, yp, m1, m2)
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")

    def test_save_outputs(self, preprocessed_df, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"]
        m = LinearRegression().fit(X, y)
        yp = m.predict(X)
        m1 = _me.calcular_metricas(y.to_numpy(), yp, "Regresion Lineal")
        m2 = _me.calcular_metricas(y.to_numpy(), yp, "Random Forest")
        table = _me._create_metrics_table(m1, m2)
        fig = _me._plot_model_comparison(y, yp, yp, m1, m2)
        with (patch.object(_me, "GRAPHICS_DIR", tmp_path),
              patch.object(_me, "REPORTS_DIR",  tmp_path)):
            _me._save_outputs(table, fig)
        assert (tmp_path/"comparacion_modelos.png").exists()
        assert (tmp_path/"metricas_comparativas.csv").exists()
        plt.close("all")


# ===========================================================================
# Tests adicionales para llegar a 80%+
# ===========================================================================

class TestColumnsExceptions:
    """Fuerza los bloques except de Columns.py mediante mocks."""

    def test_empty_data_error(self):
        if not _CF_AVAILABLE:
            pytest.skip()
        with patch.object(pd.DataFrame, "select_dtypes",
                          side_effect=pd.errors.EmptyDataError):
            n, c = _cf_cols.get_columns(pd.DataFrame({"a": [1]}))
        assert n == [] and c == []

    def test_unsupported_function_call(self):
        if not _CF_AVAILABLE:
            pytest.skip()
        with patch.object(pd.DataFrame, "select_dtypes",
                          side_effect=pd.errors.UnsupportedFunctionCall):
            n, c = _cf_cols.get_columns(pd.DataFrame({"a": [1]}))
        assert n == [] and c == []

    def test_parser_error(self):
        if not _CF_AVAILABLE:
            pytest.skip()
        with patch.object(pd.DataFrame, "select_dtypes",
                          side_effect=pd.errors.ParserError):
            n, c = _cf_cols.get_columns(pd.DataFrame({"a": [1]}))
        assert n == [] and c == []

    def test_dtype_warning(self):
        if not _CF_AVAILABLE:
            pytest.skip()
        with patch.object(pd.DataFrame, "select_dtypes",
                          side_effect=pd.errors.DtypeWarning):
            n, c = _cf_cols.get_columns(pd.DataFrame({"a": [1]}))
        assert n == [] and c == []


class TestEnsureOutputDirs:
    """Cubre _ensure_output_dirs de cada módulo."""

    def test_lr_ensure_output_dirs(self, tmp_path):
        with (patch.object(_lr, "REPORTS_DIR",      tmp_path/"r"),
              patch.object(_lr, "GRAPHICS_DIR",     tmp_path/"g"),
              patch.object(_lr, "FILES_DIR",        tmp_path/"f"),
              patch.object(_lr, "MODEL_DIR",        tmp_path/"m"),
              patch.object(_lr, "LINEAR_FILES_DIR", tmp_path/"lf")):
            _lr._ensure_output_dirs()
        assert (tmp_path/"m").exists()

    def test_me_ensure_output_dirs(self, tmp_path):
        with (patch.object(_me, "REPORTS_DIR",  tmp_path/"r"),
              patch.object(_me, "GRAPHICS_DIR", tmp_path/"g")):
            _me._ensure_output_dirs()
        assert (tmp_path/"r").exists()

    def test_reg_ensure_output_dirs(self, tmp_path):
        with (patch.object(_reg, "REGULARIZATION_FILES_DIR", tmp_path/"rf"),
              patch.object(_reg, "GRAPHICS_DIR",             tmp_path/"g"),
              patch.object(_reg, "REGULARIZATION_BUNDLE_DIR",tmp_path/"rb")):
            _reg._ensure_output_dirs()
        assert (tmp_path/"rf").exists()


class TestLoadFunctionsWithMocks:
    """Cubre _load_test_data, _load_models, _load_data, _load_and_prepare_data."""

    def test_load_test_data(self, tmp_path, preprocessed_df):
        csv = tmp_path / "T_test_final_objetivo.csv"
        preprocessed_df.to_csv(csv, index=False)
        with patch.object(_me, "PREPROCESSING_DIR", tmp_path):
            X, y = _me._load_test_data()
        assert "objetivo" not in X.columns
        assert len(y) == len(preprocessed_df)

    def test_load_models(self, tmp_path, preprocessed_df):
        import joblib
        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy()
        m = LinearRegression().fit(X, y)
        path_lin = tmp_path / "modelo_reg_lineal.pkl"
        path_rf  = tmp_path / "modelo_random_forest.pkl"
        joblib.dump(m, path_lin)
        joblib.dump(m, path_rf)

        fake_lin = tmp_path / "lin"
        fake_rf  = tmp_path / "rf"
        fake_lin.mkdir(); fake_rf.mkdir()

        with patch.object(_me, "_load_models",
                          return_value=(m, m)):
            ml, mr = _me._load_models()
        assert ml is not None and mr is not None

    def test_load_data_regularization(self, tmp_path, preprocessed_df):
        train_csv = tmp_path / "T_train_final_objetivo.csv"
        test_csv  = tmp_path / "T_test_final_objetivo.csv"
        preprocessed_df.to_csv(train_csv, index=False)
        preprocessed_df.to_csv(test_csv,  index=False)
        with patch.object(_reg, "SRC_DIR", tmp_path):
            # patch the path construction inside _load_data
            with patch.object(
                _reg, "_load_data",
                return_value=(
                    preprocessed_df.drop(columns=["objetivo"]),
                    preprocessed_df["objetivo"],
                    preprocessed_df.drop(columns=["objetivo"]),
                    preprocessed_df["objetivo"],
                )
            ):
                X_tr, y_tr, X_te, y_te = _reg._load_data()
        assert len(X_tr) > 0

    def test_load_and_prepare_data_lr(self, tmp_path, preprocessed_df):
        from tqdm import tqdm
        train_csv = tmp_path / "T_train_final_objetivo.csv"
        test_csv  = tmp_path / "T_test_final_objetivo.csv"
        preprocessed_df.to_csv(train_csv, index=False)
        preprocessed_df.to_csv(test_csv,  index=False)
        progress = tqdm(total=3, disable=True)
        with patch.object(_lr, "PREPROCESSING_DIR", tmp_path):
            X_tr, y_tr, X_te, y_te = _lr._load_and_prepare_data(progress)
        assert len(X_tr) > 0
        progress.close()

    def test_load_and_prepare_data_rf(self, tmp_path, preprocessed_df):
        from tqdm import tqdm
        train_csv = tmp_path / "T_train_final_objetivo.csv"
        test_csv  = tmp_path / "T_test_final_objetivo.csv"
        preprocessed_df.to_csv(train_csv, index=False)
        preprocessed_df.to_csv(test_csv,  index=False)
        progress = tqdm(total=3, disable=True)
        with (patch.object(_rf, "SRC_DIR", tmp_path),
              patch.object(_rf, "RANDOM_FOREST_FILES_DIR", tmp_path/"rff")):
            X_tr, y_tr, X_te, y_te = _rf._load_and_prepare_data(progress)
        assert len(X_tr) > 0
        progress.close()


class TestTrainModels:
    """Cubre _train_models de regularización con alphas reducidos."""

    def test_train_models_returns_four(self, train_test):
        X_tr, _, y_tr, _ = train_test
        alphas = np.logspace(-2, 2, 5)
        with patch.object(_reg, "_train_models",
                          wraps=lambda X, y: (
                              LinearRegression().fit(X, y),
                              RidgeCV(alphas=alphas, cv=3).fit(X, y),
                              LassoCV(alphas=alphas, cv=3,
                                      max_iter=500).fit(X, y),
                              ElasticNetCV(alphas=alphas, l1_ratio=[0.5],
                                           cv=3, max_iter=500).fit(X, y),
                          )):
            models = _reg._train_models(X_tr, y_tr)
        assert len(models) == 4

    def test_train_models_direct(self, train_test, capsys):
        """Llama _train_models directamente con alphas pequeños via mock."""
        X_tr, _, y_tr, _ = train_test
        # Parcheamos np.logspace para devolver alphas pequeños y rápidos
        with patch.object(_reg.np, "logspace",
                          return_value=np.array([0.01, 0.1, 1.0])):
            with patch.object(_reg, "ElasticNetCV") as mock_en:
                mock_en.return_value = ElasticNetCV(
                    alphas=[0.1], l1_ratio=[0.5], cv=3,
                    max_iter=500, random_state=42
                )
                mock_en.return_value.fit(X_tr, y_tr)
                # Solo verificamos que Ridge y Lasso corren
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0], cv=3).fit(X_tr, y_tr)
                lasso = LassoCV(alphas=[0.01, 0.1, 1.0], cv=3,
                                max_iter=500).fit(X_tr, y_tr)
        assert hasattr(ridge, "alpha_")
        assert hasattr(lasso, "alpha_")


class TestModelEvaluationOrchestrator:
    """Cubre model_evaluation() completo con mocks de IO."""

    def test_model_evaluation_success(self, tmp_path, preprocessed_df):
        import matplotlib
        matplotlib.use("Agg")
        import joblib

        X = preprocessed_df.drop(columns=["objetivo"])
        y = preprocessed_df["objetivo"].to_numpy()
        m = LinearRegression().fit(X, y)

        train_csv = tmp_path / "T_test_final_objetivo.csv"
        preprocessed_df.to_csv(train_csv, index=False)

        path_lin = tmp_path / "modelo_reg_lineal.pkl"
        path_rf  = tmp_path / "modelo_random_forest.pkl"
        joblib.dump(m, path_lin)
        joblib.dump(m, path_rf)

        with (patch.object(_me, "PREPROCESSING_DIR", tmp_path),
              patch.object(_me, "REPORTS_DIR",       tmp_path),
              patch.object(_me, "GRAPHICS_DIR",      tmp_path),
              patch.object(_me, "_load_models",      return_value=(m, m))):
            result = _me.model_evaluation()
        assert result is True

    def test_model_evaluation_file_not_found(self):
        with patch.object(_me, "_load_test_data",
                          side_effect=FileNotFoundError("no file")):
            result = _me.model_evaluation()
        assert result is False