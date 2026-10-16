# ANEXO 8. RESULTADOS COMPARATIVOS SOBRE LA POBLACIÓN COMÚN DE EVALUACIÓN

Base: `codex/poblacion-comun-demanda-depurada`, commit `629e36f` (incluye ABC actualizado). Rama de trabajo: `codex/cierre-anexo8-mae-only`.

## Poblaciones y fuentes

La población común contiene exactamente **3444 observaciones y 129 series sucursal-producto**. Los cuatro modelos y las tres referencias se evalúan sobre idénticas claves sucursal-producto-semana y los mismos valores reales. Se validan `resultados/semanal/predicciones_comunes.csv`, `comparacion_modelos_comun.csv` y `metricas_por_serie.csv` sin modificarlos.

La población propia se conserva en una tabla separada, copiada byte a byte desde `resultados/semanal/metricas_cobertura_propia.csv` y contrastada con las predicciones originales: Regresión Lineal, Random Forest y XGBoost cubren 3940 observaciones/156 series; ARIMA, 3444/129. No se mezclan estas poblaciones en la tabla comparativa.

## Líneas base

- **Predictor cero:** predice 0 para toda observación.
- **Persistencia:** predice el último valor semanal observado antes de la semana objetivo, incluidos ceros. Puede usar valores de semanas anteriores de TEST ya observadas, nunca la semana objetivo ni futuras. Se reutilizan las predicciones existentes de Cero y Naive_ultimo y se comprueba su definición contra el calendario semanal.
- **Media histórica:** para cada serie s, `media_train_s = sum(cantidad_s,u para u en TRAIN_s) / n_s`. Para cada semana t de TEST, `y_pred_s,t = media_train_s`. Incluye todas las semanas TRAIN disponibles, incluidos ceros. Es una media por serie, fija durante TEST; no utiliza ni actualiza con datos TEST. Si falta historial de una serie común, la ejecución se detiene sin imputación.

El corte es **2026-01-01**. Las semanas son W-SUN, etiquetadas por domingo. TRAIN exige `semana + 1 día <= corte`; TEST exige `semana - 6 días >= corte`. Se excluye la semana que cruza el corte. La media usa `dataset_semanal.csv` completo anterior al corte, no el subconjunto ML que elimina filas sin rezagos suficientes. Se verifica que el subconjunto TRAIN ML coincide con ese calendario en sus claves y cantidades.

## Métricas e interpretación

Se calculan localmente MAE = media de errores absolutos, RMSE = raíz de la media de errores cuadrados y R² = 1 - SSE/SST. El MAE medio por serie es la media no ponderada de los 129 MAE individuales. La desviación estándar del MAE por serie es muestral (`ddof=1`): refleja heterogeneidad entre series, no variabilidad de reentrenamientos ni un intervalo de confianza del MAE agregado.

Los CSV detallados mantienen precisión completa. Sólo `tabla_anexo8.csv` presenta cuatro decimales. El predictor cero puede tener MAE reducido por la abundancia de ceros, pero RMSE mayor y R² negativo; no se declara superioridad a partir de una métrica aislada.

ARIMA conserva origen fijo multi-step, mientras ML y Persistencia utilizan historia semanal observada y Media histórica permanece fija. La igualdad de población no elimina estas diferencias de protocolo. El periodo TEST ya había sido examinado: no es validación externa nueva.

**No se reentrenó ni recalibró ningún modelo. Las líneas base no participaron en la selección de los cuatro modelos principales.** Se conservan datasets, predicciones, hiperparámetros, particiones, análisis históricos, ABC, Streamlit y pruebas de significancia.

## Reproducción

Desde la raíz del repositorio:

```bash
python scripts/generar_anexo8.py
python -m unittest test_anexo8 -v
```

El script valida los artefactos antes de escribir, implementa localmente las métricas y verifica por SHA256 que los archivos experimentales previos permanecen intactos.

- `modelos_poblacion_comun.csv`: cuatro modelos, MAE/RMSE/R², MAE medio y DE por serie, observaciones y series.
- `lineas_base_poblacion_comun.csv`: las tres referencias, mismas columnas y población.
- `modelos_poblacion_propia.csv`: copia exacta de las métricas de cobertura propia.
- `predicciones_media_historica.csv`: 3444 claves comunes, y_real, y_pred y media_train.
- `tabla_anexo8.csv`: tabla compacta de siete filas para la tesis.

Los manifiestos históricos conservan el hash del notebook ABC anterior a su actualización autorizada; no se modifican esas instantáneas. Se realizan directamente las comprobaciones numéricas pertinentes sobre la evidencia vigente.
