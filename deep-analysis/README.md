# Deep Analysis

Espacio de trabajo para validar hipótesis de complejidad y estimar qué variables influyen en `TraCosto` usando p-values.

## Entorno local
- Python local fijado con `pyenv`: `3.11.9` (archivo `.python-version`).
- Entorno virtual: `.venv`.

## Ejecución (script)
Desde la raíz del proyecto:

```bash
/home/uplaph/Projects/Tramites-y-servicios/deep-analysis/.venv/bin/python \
  /home/uplaph/Projects/Tramites-y-servicios/deep-analysis/src/run_deep_analysis.py
```

## Notebook reproducible
Notebook principal:
- `deep_analysis_hypothesis_cost.ipynb`

Ejecución automática (guarda resultados y gráficas en el mismo notebook):

```bash
cd /home/uplaph/Projects/Tramites-y-servicios/deep-analysis
JUPYTER_CONFIG_DIR=$PWD/.jupyter \
JUPYTER_DATA_DIR=$PWD/.local_share/jupyter \
JUPYTER_RUNTIME_DIR=$PWD/.cache \
MPLCONFIGDIR=$PWD/.mplconfig \
/home/uplaph/Projects/Tramites-y-servicios/deep-analysis/.venv/bin/jupyter nbconvert \
  --to notebook --execute deep_analysis_hypothesis_cost.ipynb --inplace
```

## Salidas
Los resultados se escriben en `outputs/`:
- `deep_analysis_report.md`
- `hypothesis_correlations.csv`
- `pca_loadings.csv`
- `tiempo_ols_hc3.csv`
- `costo_logit_hc3.csv`
- `notebook_hypothesis_results.csv`
- `notebook_pca_loadings.csv`
- `notebook_cost_logit_table.csv`
