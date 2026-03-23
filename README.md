# Trámites y Servicios: análisis de complejidad

Proyecto de análisis de datos para estudiar trámites gubernamentales y avanzar hacia una **clasificación por nivel de complejidad**.

## Objetivo
Este repositorio concentra:
- Integración y limpieza de datos de trámites.
- Análisis exploratorio inicial.
- Validación estadística de hipótesis sobre complejidad.
- Identificación de variables asociadas al costo (`TraCosto`).

## Estado actual
- `analysis.ipynb`: notebook original del estudiante (EDA y limpieza).
- `deep-analysis/`: entorno y análisis reproducible de hipótesis + regresión logística de costo.

## Estructura del repositorio
```text
.
├── Effor_index_MASTER.csv
├── visitas_totales.csv
├── visitas_ruts.csv
├── analysis.ipynb
├── revision_notebook_complejidad.md
└── deep-analysis/
    ├── .python-version
    ├── requirements.txt
    ├── deep_analysis_hypothesis_cost.ipynb
    ├── src/
    │   └── run_deep_analysis.py
    └── outputs/
        ├── deep_analysis_report.md
        ├── hypothesis_correlations.csv
        ├── pca_loadings.csv
        ├── tiempo_ols_hc3.csv
        ├── costo_logit_hc3.csv
        ├── notebook_hypothesis_results.csv
        ├── notebook_pca_loadings.csv
        └── notebook_cost_logit_table.csv
```

## Datos
Fuentes principales:
- `Effor_index_MASTER.csv` (base maestra de trámites).
- `visitas_totales.csv` (visitas totales por `Idtram`).
- `visitas_ruts.csv` (visitas en portal RUTS por `Idtram`).

En los análisis se realiza `merge` por `Idtram`.

## Requisitos
- `pyenv` con Python `3.11.9`.
- Entorno virtual local en `deep-analysis/.venv`.

## Configuración rápida
Desde la raíz del proyecto:

```bash
cd deep-analysis
pyenv local 3.11.9
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Cómo reproducir el análisis

### 1) Script reproducible
```bash
/home/uplaph/Projects/Tramites-y-servicios/deep-analysis/.venv/bin/python \
  /home/uplaph/Projects/Tramites-y-servicios/deep-analysis/src/run_deep_analysis.py
```

### 2) Notebook reproducible con gráficos
```bash
cd /home/uplaph/Projects/Tramites-y-servicios/deep-analysis
JUPYTER_CONFIG_DIR=$PWD/.jupyter \
JUPYTER_DATA_DIR=$PWD/.local_share/jupyter \
JUPYTER_RUNTIME_DIR=$PWD/.cache \
MPLCONFIGDIR=$PWD/.mplconfig \
/home/uplaph/Projects/Tramites-y-servicios/deep-analysis/.venv/bin/jupyter nbconvert \
  --to notebook --execute deep_analysis_hypothesis_cost.ipynb --inplace
```

## Resultados actuales (resumen)
Con el análisis estadístico en `deep-analysis_hypothesis_cost.ipynb`:
- H1 (más tiempo -> más complejidad): **soportada**.
- H2 (más requisitos y más formatos -> más complejidad): **soportada**.
- H3 (más digitalización -> menos complejidad): **no soportada en el sentido esperado** (signo opuesto).

En el modelo de costo (`TraCosto`) con p-values:
- Variables significativas: `CONTEO_NETO` y `digitalizacion_num`.
- Interpretación: asociaciones estadísticas condicionales, no causalidad.

## Siguiente fase recomendada
Para iniciar la clasificación de complejidad (siguiente etapa del proyecto):
1. Confirmar semántica de `nivel_digitalizacion` (dirección de la escala).
2. Definir etiqueta operativa de complejidad (`Baja`, `Media`, `Alta`).
3. Entrenar y comparar modelos baseline con validación estratificada.
4. Revisar casos frontera con experto de dominio.

## Notas
- `analysis.ipynb` conserva el trabajo original del estudiante.
- `deep-analysis/` concentra el trabajo reproducible para hipótesis y modelado estadístico.
