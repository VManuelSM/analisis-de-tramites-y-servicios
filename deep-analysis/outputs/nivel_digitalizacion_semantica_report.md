# Análisis semántico de `nivel_digitalizacion` (Pasos 1-6)

Fecha de ejecución: `2026-03-23 16:23:57`

## Paso 1: definición oficial fijada
- Se usó exactamente la semántica oficial proporcionada para niveles 1, 2, 3.1-3.9 y 4.1-4.3.
- Se normalizó en un diccionario versionable: `nivel_digitalizacion_dictionary.csv`.

## Paso 2: diccionario semántico explícito
- Incluye: `nivel_label`, `nivel_num`, `orden_ordinal`, descripción oficial, canal de inicio/fin estimado y manejo de presencialidad estimada.
- Se añade nota de cautela para campos operativos estimados.

## Paso 3: validación de monotonicidad
- Se probó tendencia esperada decreciente (más digitalización -> menor carga) con Spearman unidireccional.
- `log_tiempo`: rho=0.0568, p_one_sided_neg=8.868e-01, decisión=`No rechazar H0`
- `requisitos`: rho=0.1564, p_one_sided_neg=9.996e-01, decisión=`No rechazar H0`
- `formatos`: rho=0.1383, p_one_sided_neg=9.985e-01, decisión=`No rechazar H0`
- `indice_carga`: rho=0.1639, p_one_sided_neg=9.998e-01, decisión=`No rechazar H0`

## Paso 4: comparación de codificaciones A/B/C
- A: ordinal directa (semánticamente válida).
- B: ordinal invertida (semánticamente no válida, solo prueba de sensibilidad).
- C: categórica one-hot (semánticamente válida, no fuerza monotonicidad).

Comparativo principal (burden y costo) en `nivel_digitalizacion_coding_comparison.csv`.
- Mejor opción práctica para modelado: `C_categorica_onehot` (burden_adj_r2=0.0283, cost_pseudo_r2=NA).

## Paso 5: faltantes como categoría propia
- `Sin_nivel`: 209 trámites (31.43% del total).
- No se imputó `nivel_digitalizacion` en este análisis; se modeló explícitamente como categoría.

## Paso 6: decisión final sobre H3
- **H3 no soportada** con semántica oficial: no se rechaza H0 de tendencia negativa (signo observado opuesto o sin evidencia suficiente).
- Conclusión operativa: usar codificación categórica para evitar imponer una relación monotónica que no está respaldada por los datos actuales.

## Archivos generados
- `nivel_digitalizacion_dictionary.csv`
- `nivel_digitalizacion_medians_by_level.csv`
- `nivel_digitalizacion_monotonicity_tests.csv`
- `nivel_digitalizacion_coding_comparison.csv`
- `nivel_digitalizacion_category_stats.csv`
- `nivel_digitalizacion_monotonicity.png`
- `nivel_digitalizacion_burden_boxplot.png`