# Deep Analysis - Hipótesis de Complejidad y Costo

## Contexto
- Fecha de ejecución: `2026-03-23 14:32:02`
- Python: `3.11.9`
- Muestra analizada: `665` trámites

## Metodología
- Índice de complejidad: primera componente principal (PCA) de `log_tiempo`, `CONTEO_NETO`, `N_FORMATOS_FINAL` y `digitalizacion_num`.
- Contraste de hipótesis: correlación de Spearman entre el índice y cada variable (con p-value).
- Sensibilidad: regresión OLS robusta (HC3) de `log_tiempo` sobre requisitos, formatos y digitalización.
- Costo: regresión logística robusta (HC3) para `TraCosto_bin` con p-values y odds ratios.

## Resultados de Hipótesis
- Varianza explicada por PC1: `0.3465`
- Cargas PC1:
  - `log_tiempo`: `0.0563`
  - `CONTEO_NETO`: `0.6930`
  - `N_FORMATOS_FINAL`: `0.6739`
  - `digitalizacion_num`: `0.2501`

### Contrastes (Spearman)
| Hipótesis | rho | p-value | Decisión |
|---|---:|---:|---|
| H1: complejidad aumenta con más tiempo | 0.1236 | 1.406e-03 | Soportada |
| H2a: complejidad aumenta con más requisitos | 0.8062 | 2.730e-153 | Soportada |
| H2b: complejidad aumenta con más formatos | 0.7629 | 1.010e-127 | Soportada |
| H3: complejidad disminuye con mayor digitalización | 0.3881 | 2.467e-25 | No soportada (signo opuesto) |

### Sensibilidad con OLS robusto (`log_tiempo` como proxy de complejidad)
| Variable | Coeficiente | p-value |
|---|---:|---:|
| CONTEO_NETO | -0.0138 | 7.844e-01 |
| N_FORMATOS_FINAL | 0.0411 | 7.706e-01 |
| digitalizacion_num | 0.1682 | 1.181e-01 |
- R² OLS robusto: `0.0039`

## Variables que influyen en `TraCosto` (p-values)
- Pseudo R² del modelo logístico: `0.0673`
- Variables significativas (alpha=0.05):
  - `CONTEO_NETO`: coef=0.2831, OR=1.3273, p=6.493e-06
  - `digitalizacion_num`: coef=0.2580, OR=1.2944, p=5.376e-03

## Interpretación breve
- H1 (tiempo): `Soportada` (rho=0.1236).
- H2 (requisitos y formatos): H2a `Soportada`, H2b `Soportada`.
- H3 (digitalización): `No soportada (signo opuesto)`; el signo observado fue 0.3881, revisar codificación semántica de niveles.
- En costo, las asociaciones significativas fueron interpretadas como efectos condicionales (no causalidad).

## Archivos generados
- `outputs/hypothesis_correlations.csv`
- `outputs/pca_loadings.csv`
- `outputs/tiempo_ols_hc3.csv`
- `outputs/costo_logit_hc3.csv`