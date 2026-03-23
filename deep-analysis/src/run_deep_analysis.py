#!/usr/bin/env python3
"""Deep statistical analysis for complexity hypotheses and cost drivers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

ALPHA = 0.05


@dataclass
class Paths:
    project_root: Path
    deep_root: Path
    output_dir: Path


def build_paths() -> Paths:
    deep_root = Path(__file__).resolve().parents[1]
    project_root = deep_root.parent
    output_dir = deep_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Paths(project_root=project_root, deep_root=deep_root, output_dir=output_dir)


def load_data(paths: Paths) -> pd.DataFrame:
    base = pd.read_csv(paths.project_root / "Effor_index_MASTER.csv")
    visitas_totales = pd.read_csv(paths.project_root / "visitas_totales.csv")[["Idtram", "visitas_totales"]]
    visitas_ruts = pd.read_csv(paths.project_root / "visitas_ruts.csv")[["Idtram", "Visitas RUTS"]]

    df = (
        base.merge(visitas_totales, on="Idtram", how="left")
        .merge(visitas_ruts, on="Idtram", how="left")
        .dropna(subset=["Idtram"])
        .copy()
    )

    df["digitalizacion_num"] = pd.to_numeric(
        df["nivel_digitalizacion"].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False),
        errors="coerce",
    )

    df["TraCosto_bin"] = (df["TraCosto"].str.upper() == "VERDADERO").astype(int)
    df["log_tiempo"] = np.log1p(df["Tiempo_en_minutos"].clip(lower=0))
    df["log_visitas_totales"] = np.log1p(df["visitas_totales"].clip(lower=0))
    df["log_visitas_ruts"] = np.log1p(df["Visitas RUTS"].clip(lower=0))

    # Imputaciones robustas para modelos con n completo.
    num_cols = [
        "Tiempo_en_minutos",
        "CONTEO_NETO",
        "N_FORMATOS_FINAL",
        "digitalizacion_num",
        "Porcentaje_Efectividad",
        "log_tiempo",
        "log_visitas_totales",
        "log_visitas_ruts",
    ]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    df["TraPersona"] = df["TraPersona"].fillna(df["TraPersona"].mode(dropna=True)[0])

    return df


def evaluate_hypothesis(sign_expected: str, estimate: float, p_value: float) -> str:
    if pd.isna(estimate) or pd.isna(p_value):
        return "No evaluable"

    sign_ok = (estimate > 0) if sign_expected == "+" else (estimate < 0)
    if p_value < ALPHA and sign_ok:
        return "Soportada"
    if p_value < ALPHA and not sign_ok:
        return "No soportada (signo opuesto)"
    return "No concluyente"


def run_hypothesis_block(df: pd.DataFrame, paths: Paths) -> tuple[pd.DataFrame, pd.Series, float]:
    pca_features = ["log_tiempo", "CONTEO_NETO", "N_FORMATOS_FINAL", "digitalizacion_num"]
    X = df[pca_features].copy()
    Z = StandardScaler().fit_transform(X)

    pca = PCA(n_components=1, random_state=42)
    complexity_score = pca.fit_transform(Z).ravel()

    loadings = pd.Series(pca.components_[0], index=pca_features, name="loading_pc1")
    # Orientamos el componente para que mayor score signifique mayor carga de tiempo.
    if loadings["log_tiempo"] < 0:
        complexity_score = -complexity_score
        loadings = -loadings

    df_corr = pd.DataFrame({
        "variable": ["log_tiempo", "CONTEO_NETO", "N_FORMATOS_FINAL", "digitalizacion_num"],
        "hipotesis": [
            "H1: complejidad aumenta con más tiempo",
            "H2a: complejidad aumenta con más requisitos",
            "H2b: complejidad aumenta con más formatos",
            "H3: complejidad disminuye con mayor digitalización",
        ],
        "signo_esperado": ["+", "+", "+", "-"],
    })

    corrs, pvals = [], []
    for var in df_corr["variable"]:
        r, p = spearmanr(complexity_score, df[var])
        corrs.append(float(r))
        pvals.append(float(p))

    df_corr["spearman_rho"] = corrs
    df_corr["p_value"] = pvals
    df_corr["decision_alpha_0.05"] = [
        evaluate_hypothesis(signo, rho, pval)
        for signo, rho, pval in zip(df_corr["signo_esperado"], df_corr["spearman_rho"], df_corr["p_value"])
    ]

    df_corr.to_csv(paths.output_dir / "hypothesis_correlations.csv", index=False)
    loadings.to_frame().to_csv(paths.output_dir / "pca_loadings.csv")

    return df_corr, loadings, float(pca.explained_variance_ratio_[0])


def run_tiempo_regression(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """Regresión robusta para sensibilidad: log_tiempo vs carga administrativa."""
    formula = "log_tiempo ~ CONTEO_NETO + N_FORMATOS_FINAL + digitalizacion_num"
    model = smf.ols(formula, data=df).fit(cov_type="HC3")

    table = model.summary2().tables[1].copy()
    p_col = [c for c in table.columns if c.startswith("P>|")][0]
    out = table[["Coef.", p_col, "[0.025", "0.975]"]].copy()
    out.columns = ["coef", "p_value", "ci_2.5", "ci_97.5"]
    out["model"] = "OLS_HC3_log_tiempo"
    out["r_squared"] = model.rsquared
    out.to_csv(paths.output_dir / "tiempo_ols_hc3.csv")
    return out


def run_cost_logit(df: pd.DataFrame, paths: Paths) -> tuple[pd.DataFrame, float]:
    formula = (
        "TraCosto_bin ~ log_tiempo + CONTEO_NETO + N_FORMATOS_FINAL + digitalizacion_num + "
        "Porcentaje_Efectividad + log_visitas_totales + log_visitas_ruts + C(TraPersona)"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False, cov_type="HC3")

    table = model.summary2().tables[1].copy()
    p_col = [c for c in table.columns if c.startswith("P>|")][0]
    out = table[["Coef.", p_col, "[0.025", "0.975]"]].copy()
    out.columns = ["coef", "p_value", "ci_2.5", "ci_97.5"]
    out["odds_ratio"] = np.exp(out["coef"])
    out["significativa_alpha_0.05"] = out["p_value"] < ALPHA
    out.to_csv(paths.output_dir / "costo_logit_hc3.csv")

    return out, float(model.prsquared)


def render_report(
    df: pd.DataFrame,
    corr_table: pd.DataFrame,
    loadings: pd.Series,
    pca_var: float,
    tiempo_tbl: pd.DataFrame,
    costo_tbl: pd.DataFrame,
    costo_pseudo_r2: float,
    paths: Paths,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(df)

    signif_cost = costo_tbl[costo_tbl["significativa_alpha_0.05"]].copy()

    lines: list[str] = []
    lines.append("# Deep Analysis - Hipótesis de Complejidad y Costo")
    lines.append("")
    lines.append("## Contexto")
    lines.append(f"- Fecha de ejecución: `{timestamp}`")
    lines.append(f"- Python: `{platform.python_version()}`")
    lines.append(f"- Muestra analizada: `{n}` trámites")
    lines.append("")
    lines.append("## Metodología")
    lines.append("- Índice de complejidad: primera componente principal (PCA) de `log_tiempo`, `CONTEO_NETO`, `N_FORMATOS_FINAL` y `digitalizacion_num`.")
    lines.append("- Contraste de hipótesis: correlación de Spearman entre el índice y cada variable (con p-value).")
    lines.append("- Sensibilidad: regresión OLS robusta (HC3) de `log_tiempo` sobre requisitos, formatos y digitalización.")
    lines.append("- Costo: regresión logística robusta (HC3) para `TraCosto_bin` con p-values y odds ratios.")
    lines.append("")
    lines.append("## Resultados de Hipótesis")
    lines.append(f"- Varianza explicada por PC1: `{pca_var:.4f}`")
    lines.append("- Cargas PC1:")
    for var, val in loadings.items():
        lines.append(f"  - `{var}`: `{val:.4f}`")
    lines.append("")
    lines.append("### Contrastes (Spearman)")
    lines.append("| Hipótesis | rho | p-value | Decisión |")
    lines.append("|---|---:|---:|---|")
    for _, row in corr_table.iterrows():
        lines.append(
            f"| {row['hipotesis']} | {row['spearman_rho']:.4f} | {row['p_value']:.3e} | {row['decision_alpha_0.05']} |"
        )

    lines.append("")
    lines.append("### Sensibilidad con OLS robusto (`log_tiempo` como proxy de complejidad)")
    lines.append("| Variable | Coeficiente | p-value |")
    lines.append("|---|---:|---:|")
    for idx in ["CONTEO_NETO", "N_FORMATOS_FINAL", "digitalizacion_num"]:
        if idx in tiempo_tbl.index:
            lines.append(
                f"| {idx} | {tiempo_tbl.loc[idx, 'coef']:.4f} | {tiempo_tbl.loc[idx, 'p_value']:.3e} |"
            )
    if "r_squared" in tiempo_tbl.columns:
        lines.append(f"- R² OLS robusto: `{tiempo_tbl['r_squared'].iloc[0]:.4f}`")

    lines.append("")
    lines.append("## Variables que influyen en `TraCosto` (p-values)")
    lines.append(f"- Pseudo R² del modelo logístico: `{costo_pseudo_r2:.4f}`")
    lines.append("- Variables significativas (alpha=0.05):")
    if signif_cost.empty:
        lines.append("  - Ninguna.")
    else:
        for idx, row in signif_cost.iterrows():
            lines.append(
                f"  - `{idx}`: coef={row['coef']:.4f}, OR={row['odds_ratio']:.4f}, p={row['p_value']:.3e}"
            )

    lines.append("")
    lines.append("## Interpretación breve")
    h1_row = corr_table.loc[corr_table["variable"] == "log_tiempo"].iloc[0]
    h2a_row = corr_table.loc[corr_table["variable"] == "CONTEO_NETO"].iloc[0]
    h2b_row = corr_table.loc[corr_table["variable"] == "N_FORMATOS_FINAL"].iloc[0]
    h3_row = corr_table.loc[corr_table["variable"] == "digitalizacion_num"].iloc[0]

    lines.append(
        f"- H1 (tiempo): `{h1_row['decision_alpha_0.05']}` (rho={h1_row['spearman_rho']:.4f})."
    )
    lines.append(
        f"- H2 (requisitos y formatos): H2a `{h2a_row['decision_alpha_0.05']}`, H2b `{h2b_row['decision_alpha_0.05']}`."
    )
    lines.append(
        f"- H3 (digitalización): `{h3_row['decision_alpha_0.05']}`; el signo observado fue {h3_row['spearman_rho']:.4f}, revisar codificación semántica de niveles." 
    )
    lines.append("- En costo, las asociaciones significativas fueron interpretadas como efectos condicionales (no causalidad).")

    lines.append("")
    lines.append("## Archivos generados")
    lines.append("- `outputs/hypothesis_correlations.csv`")
    lines.append("- `outputs/pca_loadings.csv`")
    lines.append("- `outputs/tiempo_ols_hc3.csv`")
    lines.append("- `outputs/costo_logit_hc3.csv`")

    (paths.output_dir / "deep_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = build_paths()
    df = load_data(paths)

    corr_table, loadings, pca_var = run_hypothesis_block(df, paths)
    tiempo_tbl = run_tiempo_regression(df, paths)
    costo_tbl, costo_pseudo_r2 = run_cost_logit(df, paths)

    render_report(
        df=df,
        corr_table=corr_table,
        loadings=loadings,
        pca_var=pca_var,
        tiempo_tbl=tiempo_tbl,
        costo_tbl=costo_tbl,
        costo_pseudo_r2=costo_pseudo_r2,
        paths=paths,
    )

    print("Deep analysis completed.")
    print(f"Report: {paths.output_dir / 'deep_analysis_report.md'}")


if __name__ == "__main__":
    main()
