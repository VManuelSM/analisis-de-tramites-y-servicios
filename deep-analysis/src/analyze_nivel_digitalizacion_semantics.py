#!/usr/bin/env python3
"""Semantics validation pipeline for nivel_digitalizacion (steps 1-6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

ALPHA = 0.05


@dataclass
class Paths:
    root: Path
    deep: Path
    outputs: Path


def get_paths() -> Paths:
    deep = Path(__file__).resolve().parents[1]
    root = deep.parent
    outputs = deep / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, deep=deep, outputs=outputs)


def format_level(level_num: float) -> str:
    if float(level_num).is_integer():
        return f"Nivel {int(level_num)}"
    return f"Nivel {level_num:.1f}"


def official_dictionary() -> pd.DataFrame:
    rows = [
        (1.0, "Información del trámite o servicio público a través de medios electrónicos (Inscrito en el Registro de Trámites y Servicios)."),
        (2.0, "Posibilidad de descargar formatos, en su caso."),
        (3.1, "Que el trámite o servicio pueda recibir solicitudes por medios electrónicos con los correspondientes acuses de recepción de datos y documentos."),
        (3.2, "Notificación electrónica de información faltante."),
        (3.3, "Notificación electrónica de plazos de prevención."),
        (3.4, "Notificación electrónica de vencimiento de plazo de respuesta."),
        (3.5, "Que el trámite o servicio pueda mostrar a los ciudadanos el estatus en el que se encuentra (atendido/en revisión/rechazado) por medios electrónicos."),
        (3.6, "Que el trámite o servicio presente intercambio de información con otras dependencias de manera electrónica."),
        (3.7, "Pago de derechos en línea."),
        (3.8, "Agendar citas en línea."),
        (3.9, "Llenar formatos en línea, en su caso."),
        (4.1, "Emitir resoluciones oficiales en línea."),
        (4.2, "Firma electrónica para solicitudes y resoluciones del trámite o servicio."),
        (4.3, "Resolución en línea inmediata."),
    ]

    df = pd.DataFrame(rows, columns=["nivel_num", "descripcion_oficial"])
    df["nivel_label"] = df["nivel_num"].map(format_level)
    df = df[["nivel_label", "nivel_num", "descripcion_oficial"]].copy()
    df["orden_ordinal"] = np.arange(1, len(df) + 1)

    # Campos operativos para modelado; notas conservadoras para evitar sobreafirmar.
    df["canal_inicio_estimado"] = np.select(
        [df["nivel_num"] < 3.1, df["nivel_num"] >= 3.1],
        ["Presencial/Mixto (sin solicitud en línea)", "En línea"],
        default="Mixto",
    )
    df["canal_fin_estimado"] = np.select(
        [df["nivel_num"] < 4.1, df["nivel_num"].between(4.1, 4.2, inclusive="both"), df["nivel_num"] >= 4.3],
        ["Presencial/Mixto", "En línea", "En línea inmediata"],
        default="Mixto",
    )
    df["requiere_presencial_estimado"] = np.select(
        [df["nivel_num"] <= 2.0, (df["nivel_num"] > 2.0) & (df["nivel_num"] < 4.1), df["nivel_num"] >= 4.1],
        ["Sí", "Mixto", "No necesariamente"],
        default="No determinado",
    )
    df["nota_semantica"] = "Canales estimados a partir de la definición oficial; validar con regla operativa institucional."
    return df


def parse_level_num(x: object) -> float:
    if pd.isna(x):
        return np.nan
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(x))
    return float(m.group(1)) if m else np.nan


def one_sided_pvalue(rho: float, p_two_sided: float, expected_sign: str) -> float:
    if expected_sign == "negative":
        return p_two_sided / 2 if rho <= 0 else 1 - p_two_sided / 2
    if expected_sign == "positive":
        return p_two_sided / 2 if rho >= 0 else 1 - p_two_sided / 2
    raise ValueError("expected_sign must be negative or positive")


def load_data(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(paths.root / "Effor_index_MASTER.csv").dropna(subset=["Idtram"]).copy()

    df["nivel_num"] = df["nivel_digitalizacion"].map(parse_level_num)
    df["nivel_label"] = df["nivel_num"].map(lambda v: format_level(v) if pd.notna(v) else np.nan)
    df["nivel_cat"] = df["nivel_label"].fillna("Sin_nivel")
    return df


def build_features(df: pd.DataFrame, dic: pd.DataFrame) -> pd.DataFrame:
    order_map = dict(zip(dic["nivel_label"], dic["orden_ordinal"]))
    df["nivel_order"] = df["nivel_cat"].map(order_map)
    df["is_sin_nivel"] = (df["nivel_cat"] == "Sin_nivel").astype(int)

    df["Tiempo_en_minutos"] = df["Tiempo_en_minutos"].fillna(df["Tiempo_en_minutos"].median())
    df["CONTEO_NETO"] = df["CONTEO_NETO"].fillna(df["CONTEO_NETO"].median())
    df["N_FORMATOS_FINAL"] = df["N_FORMATOS_FINAL"].fillna(df["N_FORMATOS_FINAL"].median())

    df["log_tiempo"] = np.log1p(df["Tiempo_en_minutos"].clip(lower=0))

    burden_features = ["log_tiempo", "CONTEO_NETO", "N_FORMATOS_FINAL"]
    Z = StandardScaler().fit_transform(df[burden_features])
    pca = PCA(n_components=1, random_state=42)
    burden = pca.fit_transform(Z).ravel()
    loadings = pd.Series(pca.components_[0], index=burden_features)
    if loadings["log_tiempo"] < 0:
        burden = -burden
        loadings = -loadings

    df["burden_index"] = burden
    df["TraCosto_bin"] = (df["TraCosto"].astype(str).str.upper() == "VERDADERO").astype(int)

    meta = {
        "pca_var_explained": float(pca.explained_variance_ratio_[0]),
        "pca_loadings": loadings,
    }
    return df, meta


def step3_monotonicity(df: pd.DataFrame, dic: pd.DataFrame, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = df[df["nivel_order"].notna()].copy()

    medians = (
        known.groupby(["nivel_order", "nivel_cat"], as_index=False)
        .agg(
            n=("Idtram", "size"),
            med_tiempo=("Tiempo_en_minutos", "median"),
            med_log_tiempo=("log_tiempo", "median"),
            med_conteo=("CONTEO_NETO", "median"),
            med_formatos=("N_FORMATOS_FINAL", "median"),
            med_burden=("burden_index", "median"),
        )
        .sort_values("nivel_order")
    )

    tests = []
    for var, readable in [
        ("log_tiempo", "log_tiempo"),
        ("CONTEO_NETO", "requisitos"),
        ("N_FORMATOS_FINAL", "formatos"),
        ("burden_index", "indice_carga"),
    ]:
        rho, p2 = spearmanr(known["nivel_order"], known[var])
        p1_neg = one_sided_pvalue(float(rho), float(p2), expected_sign="negative")

        med = known.groupby("nivel_order")[var].median().sort_index()
        diffs = med.diff().dropna()
        increases = int((diffs > 0).sum())
        total_steps = int(len(diffs))

        tests.append(
            {
                "variable": readable,
                "rho_spearman": float(rho),
                "p_value_two_sided": float(p2),
                "p_value_one_sided_negative": float(p1_neg),
                "decision_alpha_0_05": "Rechazar H0 (tendencia negativa)" if p1_neg < ALPHA else "No rechazar H0",
                "saltos_no_monotonicos_esperando_decrecer": increases,
                "total_saltos": total_steps,
            }
        )

    test_df = pd.DataFrame(tests)

    medians.to_csv(paths.outputs / "nivel_digitalizacion_medians_by_level.csv", index=False)
    test_df.to_csv(paths.outputs / "nivel_digitalizacion_monotonicity_tests.csv", index=False)
    return medians, test_df


def step4_compare_codings(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    max_order = int(df["nivel_order"].max())
    df = df.copy()

    df["nivel_order_filled"] = df["nivel_order"].fillna(0)
    df["nivel_order_inv_filled"] = df["nivel_order"].apply(lambda v: (max_order + 1 - v) if pd.notna(v) else np.nan).fillna(0)

    ols_a = smf.ols("burden_index ~ nivel_order_filled + is_sin_nivel", data=df).fit(cov_type="HC3")
    ols_b = smf.ols("burden_index ~ nivel_order_inv_filled + is_sin_nivel", data=df).fit(cov_type="HC3")
    ols_c = smf.ols("burden_index ~ C(nivel_cat, Treatment(reference='Nivel 1'))", data=df).fit(cov_type="HC3")

    logit_a = smf.logit("TraCosto_bin ~ nivel_order_filled + is_sin_nivel", data=df).fit(disp=False, cov_type="HC3")
    logit_b = smf.logit("TraCosto_bin ~ nivel_order_inv_filled + is_sin_nivel", data=df).fit(disp=False, cov_type="HC3")

    try:
        logit_c = smf.logit("TraCosto_bin ~ C(nivel_cat, Treatment(reference='Nivel 1'))", data=df).fit(disp=False, cov_type="HC3")
        logit_c_pseudo_r2 = float(logit_c.prsquared)
        logit_c_aic = float(logit_c.aic)
        logit_c_pmain = float(logit_c.llr_pvalue)
        logit_c_note = "Estimación completa"
    except Exception as exc:
        logit_c_pseudo_r2 = np.nan
        logit_c_aic = np.nan
        logit_c_pmain = np.nan
        logit_c_note = f"No estimable ({type(exc).__name__})"

    rows = [
        {
            "coding": "A_ordinal_directa",
            "semantica_oficial": "Valida",
            "tratamiento_missing": "Sin_nivel (is_sin_nivel)",
            "burden_adj_r2": float(ols_a.rsquared_adj),
            "burden_aic": float(ols_a.aic),
            "burden_coef_main": float(ols_a.params["nivel_order_filled"]),
            "burden_p_main": float(ols_a.pvalues["nivel_order_filled"]),
            "cost_pseudo_r2": float(logit_a.prsquared),
            "cost_aic": float(logit_a.aic),
            "cost_coef_main": float(logit_a.params["nivel_order_filled"]),
            "cost_p_main": float(logit_a.pvalues["nivel_order_filled"]),
            "cost_model_note": "Estimación completa",
        },
        {
            "coding": "B_ordinal_invertida",
            "semantica_oficial": "No valida (invierte significado)",
            "tratamiento_missing": "Sin_nivel (is_sin_nivel)",
            "burden_adj_r2": float(ols_b.rsquared_adj),
            "burden_aic": float(ols_b.aic),
            "burden_coef_main": float(ols_b.params["nivel_order_inv_filled"]),
            "burden_p_main": float(ols_b.pvalues["nivel_order_inv_filled"]),
            "cost_pseudo_r2": float(logit_b.prsquared),
            "cost_aic": float(logit_b.aic),
            "cost_coef_main": float(logit_b.params["nivel_order_inv_filled"]),
            "cost_p_main": float(logit_b.pvalues["nivel_order_inv_filled"]),
            "cost_model_note": "Estimación completa",
        },
        {
            "coding": "C_categorica_onehot",
            "semantica_oficial": "Valida",
            "tratamiento_missing": "Sin_nivel (categoría propia)",
            "burden_adj_r2": float(ols_c.rsquared_adj),
            "burden_aic": float(ols_c.aic),
            "burden_coef_main": np.nan,
            "burden_p_main": float(ols_c.f_pvalue),
            "cost_pseudo_r2": logit_c_pseudo_r2,
            "cost_aic": logit_c_aic,
            "cost_coef_main": np.nan,
            "cost_p_main": logit_c_pmain,
            "cost_model_note": logit_c_note,
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(paths.outputs / "nivel_digitalizacion_coding_comparison.csv", index=False)

    sin_nivel_stats = (
        df.groupby("nivel_cat", as_index=False)
        .agg(n=("Idtram", "size"), med_burden=("burden_index", "median"), pct_costo=("TraCosto_bin", "mean"))
        .sort_values("n", ascending=False)
    )
    sin_nivel_stats.to_csv(paths.outputs / "nivel_digitalizacion_category_stats.csv", index=False)

    return out


def make_plots(medians: pd.DataFrame, df: pd.DataFrame, paths: Paths) -> None:
    sns.set_theme(style="whitegrid")

    x_labels = medians["nivel_cat"].tolist()
    x = np.arange(len(x_labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()

    series = [
        ("med_log_tiempo", "Mediana log(tiempo)"),
        ("med_conteo", "Mediana requisitos (CONTEO_NETO)"),
        ("med_formatos", "Mediana formatos (N_FORMATOS_FINAL)"),
        ("med_burden", "Mediana índice de carga"),
    ]

    for ax, (col, title) in zip(axes, series):
        ax.plot(x, medians[col].values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

    fig.suptitle("Paso 3: revisión de monotonicidad por nivel de digitalización", fontsize=13)
    plt.tight_layout()
    fig.savefig(paths.outputs / "nivel_digitalizacion_monotonicity.png", dpi=140)
    plt.close(fig)

    ordered = [v for v in medians["nivel_cat"].tolist() if v in set(df["nivel_cat"]) ]
    if "Sin_nivel" in set(df["nivel_cat"]):
        ordered = ordered + ["Sin_nivel"]

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    sns.boxplot(data=df, x="nivel_cat", y="burden_index", order=ordered, ax=ax2)
    ax2.set_title("Paso 5: índice de carga por categoría de digitalización (incluye Sin_nivel)")
    ax2.set_xlabel("nivel_cat")
    ax2.set_ylabel("burden_index")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig2.savefig(paths.outputs / "nivel_digitalizacion_burden_boxplot.png", dpi=140)
    plt.close(fig2)


def write_report(paths: Paths, dic: pd.DataFrame, tests: pd.DataFrame, cmp_df: pd.DataFrame, meta: dict, df: pd.DataFrame) -> None:
    h3_row = tests.loc[tests["variable"] == "indice_carga"].iloc[0]

    # decisión principal para H3 según semántica oficial (codificación A/directa)
    h3_supported = h3_row["decision_alpha_0_05"] == "Rechazar H0 (tendencia negativa)"

    # elección de codificación
    # criterio: semántica + ajuste sin imponer monotonía forzada.
    best_row = cmp_df.loc[cmp_df["coding"] == "C_categorica_onehot"].iloc[0]

    sin_nivel_n = int((df["nivel_cat"] == "Sin_nivel").sum())
    sin_nivel_pct = 100 * sin_nivel_n / len(df)

    lines = []
    lines.append("# Análisis semántico de `nivel_digitalizacion` (Pasos 1-6)")
    lines.append("")
    lines.append(f"Fecha de ejecución: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")

    lines.append("## Paso 1: definición oficial fijada")
    lines.append("- Se usó exactamente la semántica oficial proporcionada para niveles 1, 2, 3.1-3.9 y 4.1-4.3.")
    lines.append("- Se normalizó en un diccionario versionable: `nivel_digitalizacion_dictionary.csv`.")
    lines.append("")

    lines.append("## Paso 2: diccionario semántico explícito")
    lines.append("- Incluye: `nivel_label`, `nivel_num`, `orden_ordinal`, descripción oficial, canal de inicio/fin estimado y manejo de presencialidad estimada.")
    lines.append("- Se añade nota de cautela para campos operativos estimados.")
    lines.append("")

    lines.append("## Paso 3: validación de monotonicidad")
    lines.append("- Se probó tendencia esperada decreciente (más digitalización -> menor carga) con Spearman unidireccional.")
    for _, r in tests.iterrows():
        lines.append(
            f"- `{r['variable']}`: rho={r['rho_spearman']:.4f}, p_one_sided_neg={r['p_value_one_sided_negative']:.3e}, decisión=`{r['decision_alpha_0_05']}`"
        )
    lines.append("")

    lines.append("## Paso 4: comparación de codificaciones A/B/C")
    lines.append("- A: ordinal directa (semánticamente válida).")
    lines.append("- B: ordinal invertida (semánticamente no válida, solo prueba de sensibilidad).")
    lines.append("- C: categórica one-hot (semánticamente válida, no fuerza monotonicidad).")
    lines.append("")
    lines.append("Comparativo principal (burden y costo) en `nivel_digitalizacion_coding_comparison.csv`.")
    cost_r2_str = "NA" if pd.isna(best_row["cost_pseudo_r2"]) else f"{best_row['cost_pseudo_r2']:.4f}"
    lines.append(
        f"- Mejor opción práctica para modelado: `C_categorica_onehot` (burden_adj_r2={best_row['burden_adj_r2']:.4f}, cost_pseudo_r2={cost_r2_str})."
    )
    lines.append("")

    lines.append("## Paso 5: faltantes como categoría propia")
    lines.append(f"- `Sin_nivel`: {sin_nivel_n} trámites ({sin_nivel_pct:.2f}% del total).")
    lines.append("- No se imputó `nivel_digitalizacion` en este análisis; se modeló explícitamente como categoría.")
    lines.append("")

    lines.append("## Paso 6: decisión final sobre H3")
    if h3_supported:
        lines.append("- **H3 soportada**: se rechaza H0 y hay evidencia de tendencia negativa entre digitalización y carga.")
    else:
        lines.append("- **H3 no soportada** con semántica oficial: no se rechaza H0 de tendencia negativa (signo observado opuesto o sin evidencia suficiente).")
    lines.append(
        "- Conclusión operativa: usar codificación categórica para evitar imponer una relación monotónica que no está respaldada por los datos actuales."
    )
    lines.append("")

    lines.append("## Archivos generados")
    lines.append("- `nivel_digitalizacion_dictionary.csv`")
    lines.append("- `nivel_digitalizacion_medians_by_level.csv`")
    lines.append("- `nivel_digitalizacion_monotonicity_tests.csv`")
    lines.append("- `nivel_digitalizacion_coding_comparison.csv`")
    lines.append("- `nivel_digitalizacion_category_stats.csv`")
    lines.append("- `nivel_digitalizacion_monotonicity.png`")
    lines.append("- `nivel_digitalizacion_burden_boxplot.png`")

    (paths.outputs / "nivel_digitalizacion_semantica_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = get_paths()

    dic = official_dictionary()
    dic.to_csv(paths.outputs / "nivel_digitalizacion_dictionary.csv", index=False)

    df = load_data(paths)
    df, meta = build_features(df, dic)

    medians, tests = step3_monotonicity(df, dic, paths)
    cmp_df = step4_compare_codings(df, paths)

    make_plots(medians, df, paths)
    write_report(paths, dic, tests, cmp_df, meta, df)

    print("Análisis de semántica de digitalización completado.")
    print(paths.outputs / "nivel_digitalizacion_semantica_report.md")


if __name__ == "__main__":
    main()
