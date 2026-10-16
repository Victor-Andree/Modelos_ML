# Anexo 8. Resultados comparativos sobre la población común de evaluación

Base: rama `codex/poblacion-comun-demanda-depurada`, commit `629e36f`, que incluye el ABC actualizado. Trabajo aislado en `codex/cierre-anexo8-baselines`. No se reentrenan ni recalibran modelos, se modifican hiperparámetros, se cambian predicciones existentes ni se selecciona un modelo para Streamlit. Estas líneas base no se utilizaron para entrenar o seleccionar los cuatro modelos principales.

## Revisión previa y fuentes

Se revisaron y validaron antes de generar salidas:

| Archivo | Comprobación |
| --- | --- |
| `resultados/semanal/comparacion_modelos_comun.csv` | Cuatro modelos, métricas vigentes confirmadas con tolerancia absoluta 1e-12; 3444 observaciones y 129 series cada uno. |
| `resultados/semanal/metricas_cobertura_propia.csv` | Valores contrastados con sus predicciones propias; copia byte a byte en este directorio. ML: 3940/156; ARIMA: 3444/129. |
| `resultados/semanal/metricas_por_serie.csv` | MAE de 129 series por modelo contrastado con las predicciones comunes. |
| `resultados/semanal/predicciones_comunes.csv` | Claves únicas sucursal-producto-semana, mismas claves y mismos y_real entre los cuatro modelos, TEST estricto. |
| `resultados/semanal/dataset_semanal.csv` | Calendario W-SUN completo: fuente de las medias y escalas TRAIN; incluye semanas cero. |
| `resultados/semanal/dataset_train_ml.csv` | 15170 filas anteriores al corte, cantidades iguales a las del calendario. No se utiliza para calcular medias: omite observaciones iniciales sin features completas. |
| `resultados/extension_intermitente/comparacion_extension.csv` | MAE/RMSE/R2/MASE de los cuatro modelos consistentes con recálculo. |
| `resultados/extension_intermitente/baselines_intermitentes.csv` | Métricas de Cero y Naive_ultimo confirmadas; Croston/SBA/TSB excluidos de este anexo. |
| `resultados/extension_intermitente/escalas_train.csv` | Igualdad con las escalas recalculadas exclusivamente con TRAIN mediante la función vigente. |

Las predicciones ya guardadas de Cero y Naive_ultimo se leen de `resultados/extension_intermitente/predicciones_comparacion_comun.csv`; no se regeneran. Se verifica que Naive_ultimo coincide con el valor previo del calendario semanal y que Cero predice exactamente cero.

## Definición de las referencias

- **Predictor cero (Cero):** y_pred = 0 en cada semana.
- **Persistencia (Naive_ultimo):** último valor semanal observado disponible antes de la semana objetivo, incluidos ceros. Puede incorporar semanas TEST anteriores ya observadas; no la semana objetivo ni futuras.
- **Media histórica (Media_historica):** para cada serie s, `media_train_s = (1 / n_s) * sum(cantidad_s,u para u en TRAIN_s)`. Para toda semana t de TEST, `y_pred_s,t = media_train_s`. Se incluyen todas las semanas de TRAIN, incluidos ceros; no se utiliza una media global, ni se actualiza durante TEST. Una serie común sin historial detiene la ejecución, sin imputación.

El corte es **2026-01-01**. La etiqueta semanal corresponde al domingo W-SUN. TRAIN sólo contiene semanas cuyo extremo exclusivo `semana + 1 día <= corte`; para TEST se comprueba `semana - 6 días >= corte`. La semana que cruza el corte queda excluida. En este calendario, TRAIN termina el 2025-12-28. El promedio se calcula antes del corte, sin usar observaciones de TEST.

## Métricas y dispersión

MAE, RMSE y R² se calculan sobre las **3444 observaciones de 129 series**. Las métricas propias están en una tabla separada; no se mezclan con la comparación común. El CSV de población propia conserva los valores y columnas originales.

Se reutiliza exactamente `metricas` de `src/extension_intermitente.py` y las escalas guardadas. Para cada serie s, `q_s = mean(abs(y_s,u - y_s,u-1))` sobre pares consecutivos de TRAIN (rezago 1). `MASE_s = MAE_s / q_s`; el MASE reportado es la **media aritmética no ponderada de MASE_s de las series con escala válida**, no una razón entre dos promedios globales. La función vigente requiere escala absoluta y cuadrática positivas. Escalas nulas o no disponibles se excluyen sólo de MASE; no se reemplazan por epsilon ni se eliminan sus observaciones de MAE/RMSE/R².

Se verificaron **92 series con escala válida y 2418 observaciones asociadas** para las siete filas. Las otras 37 series permanecen en las métricas no escaladas. Por ello un MASE menor que 1 describe la relación con la variabilidad TRAIN de ese subconjunto y no prueba superioridad frente a otra referencia sobre todo TEST.

`MAE_medio_por_serie` asigna igual peso a cada una de las 129 series. `DE_MAE_por_serie` es la desviación estándar muestral (`ddof=1`) de esos 129 MAE: expresa heterogeneidad entre series, no variabilidad de reentrenamientos ni incertidumbre del MAE semanal agregado. Para los cuatro modelos se incluye el intervalo aproximado `media ± 1.96 * DE / sqrt(129)`. Es descriptivo: la independencia entre series y la aproximación normal no están garantizadas, porque productos/sucursales pueden compartir shocks. No modifica Friedman/Wilcoxon.

ARIMA conserva origen fijo multi-step; los modelos ML y Persistencia utilizan historia observada semanalmente, y Media histórica mantiene un promedio TRAIN fijo. Compartir población no elimina las diferencias de información y protocolo. TEST ya fue examinado antes; no es una evaluación externa nueva. Un menor MAE aislado no identifica un ganador: Cero obtiene MAE bajo con RMSE mayor y R² negativo.

## Archivos y reproducción

Desde la raíz del repositorio: `python scripts/generar_anexo8.py`. Pruebas: `python -m unittest test_anexo8 -v`.

- `modelos_poblacion_comun.csv`: cuatro modelos, métricas, media/DE por serie, IC aproximado y cobertura MASE.
- `lineas_base_poblacion_comun.csv`: exclusivamente las tres referencias solicitadas, con dispersión y cobertura MASE.
- `modelos_poblacion_propia.csv`: copia exacta de la tabla original de cobertura propia.
- `predicciones_media_historica.csv`: 3444 claves comunes, y_real, y_pred y media_train.
- `tabla_anexo8.csv`: siete filas compactas, cuatro decimales sólo para presentación. Los demás CSV conservan precisión completa.
- `manifiesto_anexo8.json`: versiones y SHA256 de evidencia original y salidas. La generación comprueba que no se altera ningún archivo de datasets, código de modelos, notebooks o resultados anteriores. El hash del script normaliza únicamente finales de línea LF/CRLF.

Los validadores históricos de integridad completa continúan registrando el hash del notebook ABC previo a su actualización autorizada; este anexo no modifica esos manifiestos ni sus resultados. Sus comprobaciones numéricas pertinentes se realizan directamente desde los artefactos vigentes.
