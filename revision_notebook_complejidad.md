# Revisión técnica de `analysis.ipynb` y ruta para clasificación de complejidad

## 1) Resumen ejecutivo
El notebook **sí construye una base integrada y limpia**, y realiza un EDA inicial con nulos, descriptivos y outliers. Sin embargo, **todavía no entra a modelado** ni define formalmente la variable objetivo de complejidad. En su estado actual, el trabajo está en fase de preparación de datos.

## 2) Qué hizo exactamente el estudiante

### 2.1 Carga de datos
- Importa librerías (`pandas`, `numpy`, `seaborn`, `matplotlib`, `scipy`).
- Lee 3 fuentes desde GitHub (raw):
  - `Effor_index_MASTER.csv` (base principal)
  - `visitas_totales.csv`
  - `visitas_ruts.csv`
- Hace dos `merge` por `Idtram` para agregar:
  - `visitas_totales`
  - `Visitas RUTS`
- Resultado tras merge: **(666, 16)**.

### 2.2 Diccionario y revisión de calidad
- Construye tabla con tipo de dato, no nulos y nulos por columna.
- Detecta 1 fila con `Idtram` nulo.
- Cuantifica nulos relevantes:
  - `TraPersona`: **3.0%**
  - `Tiempo_en_minutos`: **21.17%**
  - `nivel_digitalizacion`: **31.53%**
  - `visitas_totales`: **12.61%**
  - `Visitas RUTS`: **11.26%**

### 2.3 EDA descriptivo
- Ejecuta `describe()` numérico y categórico.
- Revisa casos máximos por variable clave (`Tiempo_en_minutos`, `N_FORMATOS_FINAL`, `CONTEO_NETO`, etc.).
- Hallazgos visibles en los outputs:
  - `Tiempo_en_minutos` muy asimétrico (máx = **518400** min).
  - `visitas_totales` máx = **374832**.
  - `Visitas RUTS` máx = **176203**.

### 2.4 Distribución de tiempo (KDE)
- Toma `Tiempo_en_minutos > 0`, aplica `log10`, ajusta `gaussian_kde` y grafica densidad.
- Esto lo hace **antes y después** de imputar para comparar cambios de distribución.

### 2.5 Limpieza e imputación
- Elimina la fila con `Idtram` nulo.
- Imputa nulos con:
  - **Media** para numéricas: `Tiempo_en_minutos`, `visitas_totales`, `Visitas RUTS`.
  - **Moda** para categóricas: `nivel_digitalizacion`, `TraPersona`.
- Estado final: **665 filas, 0 nulos en todas las columnas**.

### 2.6 Outliers (regla de Tukey)
Evalúa outliers superiores (Q3 + 1.5 IQR) y extremos (Q3 + 3 IQR):
- `CONTEO_NETO`: 49 outliers (**7.37%**), 16 extremos (**2.41%**)
- `N_FORMATOS_FINAL`: 19 (**2.86%**), 3 extremos (**0.45%**)
- `visitas_totales`: 37 (**5.56%**), 27 extremos (**4.06%**)
- `Visitas RUTS`: 46 (**6.92%**), 32 extremos (**4.81%**)

## 3) Conceptos que podrían estar "opacos" y cómo leerlos

- **KDE (Kernel Density Estimation)**: estimación suave de la distribución. Sirve para ver forma (picos, colas) sin depender de bins de histograma.
- **`log10` en tiempo**: comprime valores muy grandes para que la gráfica no quede dominada por extremos.
- **Regla de Tukey (IQR)**: criterio robusto para detectar valores inusualmente altos/bajos respecto a la masa central.
- **Imputación media/moda**: rellena faltantes para completar dataset, pero puede distorsionar variables muy sesgadas (ej. tiempo).

## 4) Diagnóstico técnico (fortalezas y riesgos)

### Fortalezas
- Integración de fuentes por `Idtram` correctamente (sin explosión de filas).
- EDA inicial ordenado y con narrativa.
- Limpieza explícita y verificación final de nulos.

### Riesgos / áreas de mejora
- Aún **no existe variable objetivo de complejidad** (no hay etiqueta `y`).
- Imputación por **media** en variables muy sesgadas puede sesgar resultados.
- Se detectan outliers, pero **no se define tratamiento** posterior (winsorizar, transformar, conservar con modelo robusto, etc.).
- Se mezclan columnas que pueden medir **demanda** (`visitas`) más que complejidad intrínseca.
- Hay detalles de implementación:
  - `plt.figure` aparece sin paréntesis (`plt.figure()` sería lo correcto).
  - `kagglehub` importado pero no usado.
  - Celda duplicada para máximo de `visitas_totales`.
- El diccionario dice `nivel_digitalizacion` entero, pero en datos es categórica tipo `Nivel 1`, `Nivel 4.1`, etc.

## 5) Cómo continuar para clasificar trámites por complejidad

## Paso A: Definir complejidad (regla de negocio)
Acordar con el equipo una definición operativa. Recomendación:
- Complejidad aumenta con: más tiempo, más requisitos (`CONTEO_NETO`), más formatos.
- Complejidad disminuye con mayor digitalización.
- Opcional: incluir costo y resoluciones negativas como factores secundarios.

## Paso B: Crear etiqueta inicial (sin esperar etiquetado manual completo)
Construir un **score de complejidad** y luego discretizar en clases (`Baja`, `Media`, `Alta`).
Ejemplo base:
\[
S = w_1 z(\log(1+Tiempo)) + w_2 z(CONTEO\_NETO) + w_3 z(N\_FORMATOS\_FINAL) - w_4 z(Digitalizacion\_num)
\]
- Convertir `nivel_digitalizacion` a escala numérica ordinal (1, 2, 3.1, ...).
- Definir pesos con experto (o iguales al inicio).
- Cortar por terciles/quintiles para clase.

## Paso C: Entrenar baseline de clasificación
- Modelos iniciales recomendados:
  - `LogisticRegression` multiclase (interpretable)
  - `RandomForest` o `XGBoost` (no lineal)
- Métricas: `f1_macro`, matriz de confusión, balanced accuracy.
- Validación: cross-validation estratificada.

## Paso D: Validación con experto de dominio
- Revisar casos frontera (top falsos positivos/negativos).
- Ajustar pesos/reglas de etiqueta si hay incoherencias semánticas.

## Paso E: Entregable reproducible
- Separar en scripts/notebooks:
  - `01_eda_limpieza.ipynb`
  - `02_etiquetado_complejidad.ipynb`
  - `03_modelado_clasificacion.ipynb`
- Congelar dataset analítico: `data/processed/tramites_model.csv`.

## 6) Propuesta de trabajo conjunto (tú + estudiante)

- **Estudiante (Física)**:
  - Implementa pipeline reproducible de limpieza/feature engineering.
  - Construye y compara 2-3 modelos baseline.
- **Tú (maestría IA)**:
  - Diriges definición formal de `complejidad` y criterios de evaluación.
  - Validas supuestos estadísticos, sesgos y calidad de etiquetas.
  - Diseñas protocolo de validación y narrativa metodológica para tesis/reporte.

Ritmo sugerido:
- Sesión 1: definición de complejidad + variables permitidas.
- Sesión 2: revisión de etiqueta inicial + primer modelo.
- Sesión 3: refinamiento + resultados finales y visualizaciones.

## 7) Siguiente sprint mínimo (práctico)
1. Crear mapeo limpio de `nivel_digitalizacion` a escala numérica.
2. Cambiar imputación de `Tiempo_en_minutos` de media a mediana (o KNN si justifican).
3. Construir `score_complejidad_v1` + clase terciles.
4. Entrenar baseline y reportar métricas.
5. Revisar 20 casos con experto para recalibrar etiqueta.

---
Si quieres, el siguiente paso puede ser que te deje armado un notebook nuevo (`02_etiquetado_complejidad.ipynb`) con el pipeline de etiqueta + primer clasificador listo para correr.
