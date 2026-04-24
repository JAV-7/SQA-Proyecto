"""
Preprocessing (Training).

Este modulo prepara datos de entrenamiento/prueba, entrena el encoder
categorico y el pipeline numerico con PCA, y exporta artefactos para
uso en produccion.

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa

V 0.0
"""

import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from tqdm import tqdm

from Common_Functions.IQR import iqr_outlier_stats

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PREPROCESSING_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SRC_DIR / "reports"
GRAPHICS_DIR = SRC_DIR / "graphics"
FILES_DIR = SRC_DIR / "files"

INPUT_DEFAULT_PATH = FILES_DIR / "retail_store_inventory_entrenamiento.csv"
REPORT_PATH = REPORTS_DIR / "01_a_preprocessing_report.txt"
MIN_COMPONENTS_FOR_3D_PCA_PLOT = 3


@dataclass(frozen=True)
class PcaPlotContext:
    """Agrupa parametros del PCA para simplificar firma de funciones."""

    explained_ratio: np.ndarray
    cumulative_ratio: np.ndarray
    k90: int
    k95: int
    k_elbow: int
    show_plots: bool


def _ensure_output_dirs() -> None:
    """Crea los directorios de salida si no existen."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)


def _save_numeric_diagnostic_plots(df_num: pd.DataFrame) -> None:
    """Genera y guarda histogramas y boxplots para columnas numéricas."""
    cols_num = list(df_num.columns)
    if not cols_num:
        return

    n = len(cols_num)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3.8 * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    for index, col in enumerate(cols_num):
        values = pd.to_numeric(df_num[col], errors="coerce").dropna()
        ax = axes[index]
        if values.empty:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.hist(values, bins=30)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")

    for index in range(len(cols_num), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Histogramas de columnas numericas", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(
        GRAPHICS_DIR / "01_a_histogramas_numericas.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    series_pairs: list[tuple[str, np.ndarray]] = []
    for col in cols_num:
        values = pd.to_numeric(df_num[col], errors="coerce").dropna().values
        if values.size > 0:
            series_pairs.append((col, values))

    if not series_pairs:
        return

    labels = [col for col, _ in series_pairs]
    values = [arr for _, arr in series_pairs]

    fig2 = plt.figure(figsize=(1.6 * len(labels) + 4, 5))
    plt.boxplot(values, vert=True, showmeans=True)
    plt.xticks(
        ticks=range(1, len(labels) + 1),
        labels=labels,
        rotation=35,
        ha="right",
    )
    plt.ylabel("Valor")
    plt.title("Boxplots de columnas numericas")
    plt.tight_layout()
    fig2.savefig(
        GRAPHICS_DIR / "01_a_boxplots_numericas.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig2)


def _save_pca_plots(
    t_train_df: pd.DataFrame,
    y_train: pd.Series,
    context: PcaPlotContext,
) -> None:
    """Genera y guarda gráficos relacionados con PCA."""
    explained_ratio = context.explained_ratio
    cumulative_ratio = context.cumulative_ratio
    ks = np.arange(1, len(explained_ratio) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(ks, explained_ratio, marker="o")
    axs[0].set_xlabel("Componente principal")
    axs[0].set_ylabel("Varianza explicada")
    axs[0].set_title("Scree plot (codo)")
    axs[0].grid(True, linewidth=0.4, alpha=0.5)
    axs[0].axvline(context.k_elbow, linestyle="--",
                   label=f"codo~{context.k_elbow}")
    axs[0].legend(frameon=False)

    axs[1].plot(ks, cumulative_ratio, marker="o")
    axs[1].set_xlabel("Componente principal")
    axs[1].set_ylabel("Varianza explicada acumulada")
    axs[1].set_title("Varianza acumulada")
    axs[1].grid(True, linewidth=0.4, alpha=0.5)
    axs[1].axhline(0.90, linestyle="--")
    axs[1].axvline(context.k90, linestyle="--")
    axs[1].axhline(0.95, linestyle="--")
    axs[1].axvline(context.k95, linestyle="--")

    fig.tight_layout()
    fig.savefig(
        GRAPHICS_DIR / "01_a_pca_codo_varianza.png",
        dpi=150,
        bbox_inches="tight",
    )
    if context.show_plots:
        plt.show()
    plt.close(fig)

    df_plot2 = t_train_df.iloc[:, :2].copy()
    df_plot2["objetivo"] = y_train.loc[t_train_df.index].astype(str)
    fig2 = px.scatter(
        df_plot2,
        x="PC1",
        y="PC2",
        color="objetivo",
        opacity=0.85,
        title="PC1 vs PC2 por objetivo (train)",
        height=550,
    )
    fig2.update_traces(marker={"size": 6})
    fig2.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    fig2.write_html(GRAPHICS_DIR / "01_a_pca_2d_train.html")
    if context.show_plots:
        fig2.show()

    if t_train_df.shape[1] >= MIN_COMPONENTS_FOR_3D_PCA_PLOT:
        df_plot3 = t_train_df.iloc[:, :3].copy()
        df_plot3["objetivo"] = y_train.loc[t_train_df.index].astype(str)
        fig3 = px.scatter_3d(
            df_plot3,
            x="PC1",
            y="PC2",
            z="PC3",
            color="objetivo",
            opacity=0.85,
            title="PC1-PC2-PC3 por objetivo (train)",
            height=600,
        )
        fig3.update_traces(marker={"size": 5})
        fig3.update_layout(
            scene={
                "xaxis_title": "PC1",
                "yaxis_title": "PC2",
                "zaxis_title": "PC3",
            }
        )
        fig3.write_html(GRAPHICS_DIR / "01_a_pca_3d_train.html")


def preprocessing(  # noqa: PLR0914, PLR0915
    input_csv_path: str | Path | None = None,
    test_size: float = 0.25,
    random_state: int = 0,
    pca_components: int = 5,
    show_plots: bool = False,
) -> bool:
    """Ejecuta el preprocesamiento de entrenamiento y retorna True/False."""
    _ensure_output_dirs()
    progress = tqdm(total=7, desc="Preprocessing Train", unit="paso")

    input_path = Path(input_csv_path) if input_csv_path else INPUT_DEFAULT_PATH

    try:
        progress.set_postfix_str("Cargando datos")
        df = pd.read_csv(input_path)

        y = df["Demand Forecast"]
        x = df.drop(columns=["Demand Forecast"])

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        progress.update(1)

        cols_num = [
            "Store ID",
            "Product ID",
            "Inventory Level",
            "Units Sold",
            "Units Ordered",
            "Price",
            "Discount",
            "Competitor Pricing",
            "date_timestamp",
        ]
        cols_cat = ["Category", "Region", "Holiday/Promotion", "Seasonality"]

        x_train_num = x_train[cols_num]
        x_test_num = x_test[cols_num]
        x_train_cat = x_train[cols_cat]
        x_test_cat = x_test[cols_cat]
        progress.set_postfix_str("Graficos diagnostico")

        _save_numeric_diagnostic_plots(x_train_num)
        progress.update(1)

        cols_onehot = [
            "Category", "Region", "Holiday/Promotion", "Seasonality"
            ]
        preprocessor_cat = ColumnTransformer(
            transformers=[
                (
                    "onehot",
                    OneHotEncoder(
                        sparse_output=False, drop=None, handle_unknown="ignore"
                        ),
                    cols_onehot,
                )
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        preprocessor_cat.fit(x_train_cat)
        x_train_cat_proc = preprocessor_cat.transform(x_train_cat)
        x_test_cat_proc = preprocessor_cat.transform(x_test_cat)
        progress.set_postfix_str("Encoding categoricas")
        progress.update(1)

        onehot = preprocessor_cat.named_transformers_.get("onehot")
        rename_map: dict[str, str] = {}
        if onehot is not None:
            for col, cats in zip(
                cols_onehot, onehot.categories_, strict=False
                ):
                for cat in cats:
                    rename_map[f"{col}_{cat}"] = f"{col}___{cat}"

        cat_out_cols = [
            rename_map.get(col, col)
            for col in preprocessor_cat.get_feature_names_out()
        ]

        df_train_cat_encode = pd.DataFrame(
            x_train_cat_proc,
            columns=cat_out_cols,
            index=x_train_cat.index,
        )
        df_test_cat_encode = pd.DataFrame(
            x_test_cat_proc,
            columns=cat_out_cols,
            index=x_test_cat.index,
        )

        num_mean_min_cols = ["Discount"]
        num_mean_std_cols = [
            "Units Sold",
            "Units Ordered",
            "Inventory Level",
            "Price",
            "Competitor Pricing",
            "date_timestamp",
        ]

        pipe_mean_min = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler()),
            ]
        )
        pipe_mean_std = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor_num = ColumnTransformer(
            transformers=[
                ("num_mean_min", pipe_mean_min, num_mean_min_cols),
                ("num_mean_std", pipe_mean_std, num_mean_std_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        pca_pipe_full = Pipeline(
            steps=[
                ("pre", preprocessor_num),
                ("std_for_pca", StandardScaler()),
                ("pca", PCA(n_components=None, svd_solver="full",
                            random_state=random_state)),
            ],
            memory=None,
        )
        pca_pipe_full.fit(x_train_num)

        pca_full = pca_pipe_full.named_steps["pca"]
        explained_ratio_full = pca_full.explained_variance_ratio_
        cumulative_ratio_full = np.cumsum(explained_ratio_full)
        k90 = int(np.searchsorted(cumulative_ratio_full, 0.90) + 1)
        k95 = int(np.searchsorted(cumulative_ratio_full, 0.95) + 1)

        d2 = np.diff(np.diff(explained_ratio_full))
        k_elbow = int(np.argmax(-d2) + 2) if len(d2) else 1

        pca_pipe = Pipeline(
            steps=[
                ("pre", preprocessor_num),
                ("std_for_pca", StandardScaler()),
                (
                    "pca",
                    PCA(n_components=pca_components, svd_solver="full",
                        random_state=random_state),
                ),
            ],
            memory=None,
        )
        pca_pipe.fit(x_train_num)
        progress.set_postfix_str("Entrenando PCA")
        progress.update(1)

        t_train = pca_pipe.transform(x_train_num)
        t_test = pca_pipe.transform(x_test_num)

        pc_cols = [f"PC{i + 1}" for i in range(t_train.shape[1])]
        t_train_df = pd.DataFrame(
            t_train, columns=pc_cols, index=x_train_num.index
            )
        t_test_df = pd.DataFrame(
            t_test, columns=pc_cols, index=x_test_num.index
            )

        pca_context = PcaPlotContext(
            explained_ratio=explained_ratio_full,
            cumulative_ratio=cumulative_ratio_full,
            k90=k90,
            k95=k95,
            k_elbow=k_elbow,
            show_plots=show_plots,
        )
        _save_pca_plots(t_train_df=t_train_df, y_train=y_train,
                        context=pca_context)
        progress.set_postfix_str("Generando graficos PCA")
        progress.update(1)

        df_train_cat_encode = df_train_cat_encode.reindex(t_train_df.index)
        df_test_cat_encode = df_test_cat_encode.reindex(t_test_df.index)

        t_train_final = pd.concat([t_train_df, df_train_cat_encode], axis=1)
        t_test_final = pd.concat([t_test_df, df_test_cat_encode], axis=1)

        t_train_final_out = t_train_final.copy()
        t_test_final_out = t_test_final.copy()
        t_train_final_out["objetivo"] = y_train.loc[
            t_train_final.index
        ].astype(str)
        t_test_final_out["objetivo"] = y_test.loc[
            t_test_final.index
        ].astype(str)

        t_train_df.to_csv(
            PREPROCESSING_DIR / "T_train_PCA.csv", index=False
            )
        t_test_df.to_csv(
            PREPROCESSING_DIR / "T_test_PCA.csv", index=False
            )
        t_train_final.to_csv(
            PREPROCESSING_DIR / "T_train_final.csv", index=False
            )
        t_test_final.to_csv(
            PREPROCESSING_DIR / "T_test_final.csv", index=False
            )
        t_train_final_out.to_csv(
            PREPROCESSING_DIR / "T_train_final_objetivo.csv", index=False
            )
        t_test_final_out.to_csv(
            PREPROCESSING_DIR / "T_test_final_objetivo.csv", index=False
            )

        joblib.dump(
            preprocessor_cat, PREPROCESSING_DIR / "preprocessor_cat.joblib"
            )
        joblib.dump(pca_pipe, PREPROCESSING_DIR / "pca_pipe_num.joblib")

        meta = {
            "cols_num": cols_num,
            "cols_cat": cols_cat,
            "pc_cols": pc_cols,
            "cat_out_cols": list(df_train_cat_encode.columns),
        }
        with open(
            PREPROCESSING_DIR / "pca_metadata.json", "w", encoding="utf-8"
            ) as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)

        artifacts_dir = PREPROCESSING_DIR / "mi_pca"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        zip_path = artifacts_dir / "pca_artifacts_bundle.zip"

        candidates = [
            PREPROCESSING_DIR / "T_test_PCA.csv",
            PREPROCESSING_DIR / "T_test_final.csv",
            PREPROCESSING_DIR / "T_test_final_objetivo.csv",
            PREPROCESSING_DIR / "T_train_PCA.csv",
            PREPROCESSING_DIR / "T_train_final.csv",
            PREPROCESSING_DIR / "T_train_final_objetivo.csv",
            PREPROCESSING_DIR / "pca_metadata.json",
            PREPROCESSING_DIR / "pca_pipe_num.joblib",
            PREPROCESSING_DIR / "preprocessor_cat.joblib",
            GRAPHICS_DIR / "01_a_histogramas_numericas.png",
            GRAPHICS_DIR / "01_a_boxplots_numericas.png",
            GRAPHICS_DIR / "01_a_pca_codo_varianza.png",
        ]

        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zip_file:
            for path in candidates:
                if path.exists():
                    zip_file.write(path, arcname=os.path.basename(path))
        progress.set_postfix_str("Guardando artefactos")
        progress.update(1)

        diagnostico = []
        for col in cols_num:
            outliers_n, outliers_pct = iqr_outlier_stats(x_train[col])
            diagnostico.append(
                {
                    "columna": col,
                    "missing_pct": round(pd.to_numeric(x_train[col],
                                    errors="coerce").isna().mean() * 100, 2),
                    "outliers_n": outliers_n,
                    "outliers_pct": round(outliers_pct * 100, 2),
                }
            )
        diag_df = pd.DataFrame(diagnostico)
        diag_df.to_csv(
            REPORTS_DIR / "01_a_preprocessing_numeric_diagnostic.csv",
            index=False)

        with open(REPORT_PATH, "w", encoding="utf-8") as file:
            file.write("Preprocessing Report\n")
            file.write("=" * 80 + "\n")
            file.write(f"Input: {input_path}\n")
            file.write(f"Train shape: {x_train.shape}\n")
            file.write(f"Test shape: {x_test.shape}\n")
            file.write(f"PCA components selected: {pca_components}\n")
            file.write(f"k90: {k90} | k95: {k95} | k_elbow: {k_elbow}\n")
            file.write(f"T_train_final shape: {t_train_final.shape}\n")
            file.write(f"T_test_final shape: {t_test_final.shape}\n")
            file.write(f"Artifacts ZIP: {zip_path}\n")
            file.write("\nGenerated files:\n")
            file.write("- T_train_final.csv\n")
            file.write("- T_test_final.csv\n")
            file.write("- T_train_final_objetivo.csv\n")
            file.write("- T_test_final_objetivo.csv\n")
            file.write("- preprocessor_cat.joblib\n")
            file.write("- pca_pipe_num.joblib\n")
            file.write("- pca_metadata.json\n")

        progress.set_postfix_str("Completado")
        progress.update(1)

        return True
    except FileNotFoundError:
        print(f"Archivo no encontrado: {input_path}")
        return False
    except pd.errors.EmptyDataError:
        print(f"Archivo vacío: {input_path}")
        return False
    finally:
        progress.close()


if __name__ == "__main__":
    preprocessing()
