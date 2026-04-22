"""
Preprocessing Production

Estudiantes: Francisco Javier Ramos Jimenez,
             Karen Elizabeth Gonzalez Santana

Materia: Calidad de Software

Docente: Sarahi Partida Ochoa

Creditos especiales: Sofia Vanessa Noyola,
                     Sebastian Garcia-Moreno Zinchenko,
                     Mtro. Miguel Tlapa           

V 0.0 
"""

import joblib
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = BASE_DIR / "Preprocessing"
SRC_DIR = BASE_DIR.parent
REPORTS_DIR = SRC_DIR / "reports"
GRAPHICS_DIR = SRC_DIR / "graphics"
FILES_DIR = SRC_DIR / "files"

INPUT_DEFAULT_PATH = FILES_DIR / "retail_store_inventory_produccion_unknown.csv"
OUTPUT_DEFAULT_PATH = FILES_DIR / "T_new_final.csv"
REPORT_DEFAULT_PATH = REPORTS_DIR / "01_b_preprocessing_production_report.txt"
PLOT_3D_DEFAULT_PATH = GRAPHICS_DIR / "pca_3d_nuevos.html"


def _ensure_output_dirs() -> None:
    """Crea los directorios de salida si no existen."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)

def preprocessing_production(
    input_csv_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> bool:
    """Ejecuta el preprocesamiento de producción y retorna True/False."""
    _ensure_output_dirs()

    input_path = Path(input_csv_path) if input_csv_path else INPUT_DEFAULT_PATH
    output_path = Path(output_csv_path) if output_csv_path else OUTPUT_DEFAULT_PATH

    try:
        preprocessor_cat = joblib.load(PREPROCESSING_DIR / "preprocessor_cat.joblib")
        pca_pipe = joblib.load(PREPROCESSING_DIR / "pca_pipe_num.joblib")

        with open(PREPROCESSING_DIR / "pca_metadata.json", "r", encoding="utf-8") as file:
            meta = json.load(file)

        cols_num = meta["cols_num"]
        cols_cat = meta["cols_cat"]
        pc_cols = meta["pc_cols"]
        cat_out_cols = meta["cat_out_cols"]

        entrenamiento = pd.read_csv(PREPROCESSING_DIR / "T_train_final_objetivo.csv")
        entrenamiento_pca_objetivo = entrenamiento[pc_cols]

        new_df = pd.read_csv(input_path)

        x_new_cat = new_df[cols_cat]
        x_new_num = new_df[cols_num]

        x_new_cat_proc = preprocessor_cat.transform(x_new_cat)
        df_new_cat_encode = pd.DataFrame(
            x_new_cat_proc,
            columns=cat_out_cols,
            index=new_df.index,
        )

        t_new = pca_pipe.transform(x_new_num)
        t_new_df = pd.DataFrame(t_new, columns=pc_cols, index=new_df.index)

        t_new_final = pd.concat([t_new_df, df_new_cat_encode], axis=1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        t_new_final.to_csv(output_path, index=False)

        y_train = entrenamiento["objetivo"]
        df_plot2 = entrenamiento_pca_objetivo.iloc[:, :2].copy()
        df_plot2["objetivo"] = y_train.loc[entrenamiento_pca_objetivo.index].astype(str)

        fig = px.scatter(
            df_plot2,
            x="PC1",
            y="PC2",
            color="objetivo",
            opacity=0.85,
            title="PC1 vs PC2 — Train (objetivo) + Nuevos",
            height=550,
        )
        fig.update_traces(marker={"size": 6})

        new2 = t_new_df.loc[:, ["PC1", "PC2"]].copy()
        fig.add_trace(
            go.Scatter(
                x=new2["PC1"],
                y=new2["PC2"],
                mode="markers",
                name="nuevos",
                marker={"size": 9, "line": {"width": 1}},
                opacity=0.95,
                showlegend=True,
            )
        )
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
        if show_plots:
            fig.show()

        df3 = entrenamiento_pca_objetivo.iloc[:, :3].copy()
        df3["objetivo"] = y_train.loc[entrenamiento_pca_objetivo.index].astype(str)

        fig3 = px.scatter_3d(
            df3,
            x="PC1",
            y="PC2",
            z="PC3",
            color="objetivo",
            opacity=0.85,
            title="PC1–PC2–PC3 — Train (objetivo) + Nuevos",
            height=600,
        )
        fig3.update_traces(marker={"size": 5})

        new3 = t_new_df.loc[:, ["PC1", "PC2", "PC3"]].copy()
        fig3.add_trace(
            go.Scatter3d(
                x=new3["PC1"],
                y=new3["PC2"],
                z=new3["PC3"],
                mode="markers",
                name="nuevos",
                marker={"size": 6, "line": {"width": 1}},
                opacity=0.95,
                showlegend=True,
            )
        )

        fig3.update_layout(
            scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"}
        )

        if save_plots:
            fig3.write_html(PLOT_3D_DEFAULT_PATH)

        with open(REPORT_DEFAULT_PATH, "w", encoding="utf-8") as file:
            file.write("Preprocessing Production Report\n")
            file.write("=" * 80 + "\n")
            file.write(f"Input: {input_path}\n")
            file.write(f"Output CSV: {output_path}\n")
            file.write(f"Output Plot 3D: {PLOT_3D_DEFAULT_PATH if save_plots else 'No generado'}\n")
            file.write(f"Registros procesados: {len(t_new_final)}\n")
            file.write(f"Columnas finales: {list(t_new_final.columns)}\n")

        return True
    except Exception as error:
        print(f"Error en preprocessing_production: {error}")
        return False

if __name__ == "__main__":
    preprocessing_production()





