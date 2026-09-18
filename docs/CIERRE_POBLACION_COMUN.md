# Cierre técnico: target depurado y evaluación sobre población común

Se implementó y ejecutó en una copia local de `Victor-Andree/Modelos_ML`. No se modificaron Discusión, Conclusiones ni Abstract. Los archivos originales de `datasets/` son idénticos byte a byte a la copia de partida; se conservan sus hashes en `integridad_datos_originales.json`.

## Limpieza y poblaciones

- Registros iniciales: **44,389**.
- Negativos eliminados: **74**, suma **-93**; **10 productos y 13 sucursales** afectados.
- Nulos eliminados: **0**.
- Registros analíticos: **44,315**; los ceros permanecen.
- Observaciones semanales: **21,123 → 21,123**. Se modificó la cantidad de **69** claves semanales. Suma del target: **3177 → 3259**.
- Calendario de 240 series; 144 observaciones de la semana cruzada excluidas del scoring; 20 979 evaluables antes de lags.
- ML: **15 170 TRAIN**, **3 940 TEST**, 156 series en TEST. Se pierden 1 869 filas evaluables por historia insuficiente.
- ARIMA: **3 444 TEST**, 129 series. Conteos: `{'potenciales': 266, 'encontradas': 240, 'descartadas': 111, 'fallidas': 0, 'modeladas': 129}`.
- Población común final: **3 444 claves sucursal-producto-semana y 129 series por modelo**. El archivo largo `predicciones_comunes.csv` contiene **13 776 filas**, cuatro por clave.
- ML conserva 87.4112% de su cobertura propia al comparar: se excluyen 496 observaciones y 27 series de esa tabla, sin borrarlas de las predicciones propias. ARIMA conserva 100%.

## Métricas finales comunes

| Modelo | MAE | RMSE | R2 | Observaciones_evaluadas | Series_evaluadas |
| --- | --- | --- | --- | --- | --- |
| Regresión Lineal | 0.2306 | 0.5396 | 0.3219 | 3444 | 129 |
| Random Forest | 0.2270 | 0.5487 | 0.2989 | 3444 | 129 |
| XGBoost | 0.2252 | 0.5489 | 0.2984 | 3444 | 129 |
| ARIMA (Optimizado ADF/AIC) | 0.2363 | 0.5797 | 0.2175 | 3444 | 129 |

Regresión Lineal alcanza menor RMSE y mayor R²; XGBoost alcanza menor MAE en esta población y con los protocolos conservados. No se impuso ningún ganador. Con el mismo y_real, RMSE y R² ordenan necesariamente los modelos de forma inversa: no son dos evidencias independientes.

Los CSV guardan la precisión completa. Estas tablas muestran cuatro decimales. `src/evaluacion_semanal.py` calcula todas las métricas desde predicciones; no se copian valores manualmente entre tablas.

## Cambios respecto al experimento anterior

Anterior, **cobertura propia** y target neto truncado:

| Modelo | MAE | RMSE | R2 | Observaciones_evaluadas |
| --- | --- | --- | --- | --- |
| Regresión Lineal | 0.2126 | 0.5065 | 0.3265 | 3940 |
| XGBoost | 0.2056 | 0.5140 | 0.3063 | 3940 |
| Random Forest | 0.2072 | 0.5175 | 0.2969 | 3940 |
| ARIMA (Optimizado ADF/AIC) | 0.2313 | 0.5730 | 0.2103 | 3444 |

Nuevo target, todavía en **cobertura propia**:

| Modelo | MAE | RMSE | R2 | Observaciones_evaluadas | Series_evaluadas |
| --- | --- | --- | --- | --- | --- |
| Regresión Lineal | 0.2174 | 0.5150 | 0.3250 | 3940 | 156 |
| Random Forest | 0.2106 | 0.5233 | 0.3031 | 3940 | 156 |
| XGBoost | 0.2103 | 0.5238 | 0.3018 | 3940 | 156 |
| ARIMA (Optimizado ADF/AIC) | 0.2363 | 0.5797 | 0.2175 | 3444 | 129 |

La primera transición refleja la nueva definición del target y el reentrenamiento; la transición de cobertura propia nueva a población común refleja el subconjunto evaluado. Comparar directamente las métricas anteriores con las comunes mezcla ambos efectos y no demuestra una mejora o un deterioro intrínseco del algoritmo. El detalle a precisión completa está en `cambios_metricas.csv`.

La referencia ARIMA anterior procede de los CSV de esta copia (RMSE 0.5730), no del valor 0.5728 mostrado en otra ejecución local del usuario. No se combinan cifras de ejecuciones distintas.

## ARIMA y metodología preservada

Se verificó equivalencia AST de `_determinar_d` y `_optimizar_pq`, y de los métodos de búsqueda, rejillas, estimadores y codificación ML. Continúan ADF, AIC p/q 0..3; RF [100,200] árboles, profundidades [10,20], hojas [1,2]; XGBoost [100,200], learning_rate [0.05,0.1], profundidad [3,5]; semilla 42 y TimeSeriesSplit(n_splits=3) sobre semanas únicas.

Descartes ARIMA por razón:

```text
estado      motivo                   
descartada  menos_de_10_semanas_train    22
            menos_de_20_semanas          17
            menos_de_2_semanas_test      72
```

El registro individual está en `registro_series_arima.csv`. Las posibles series desaparecidas tras limpiar se auditan en `auditoria_series_limpieza.csv`.

## Validación

- 17 pruebas automáticas aprobadas: eliminación de NaN/negativos sin mutar RAW, conservación de ceros, independencia entre series, ausencia de futuro en lags/rolling, exclusión del cruce sin comprimir lags ni horizontes, folds temporales, métricas sin redondear, duplicados y targets incompatibles rechazados, intersección exacta y valores no finitos.
- Notebooks 06, 07 y 08 ejecutados completos y sin errores.
- Validador de evidencia aprobado: agregación independiente desde transacciones, target no negativo/no nulo, fechas de test desde el corte, claves idénticas de cuatro modelos, conteos/series, métricas recalculadas, auditorías y hashes.
- Modelos ML exportados entrenados sólo con TRAIN. La recarga coincide dentro de 1e-12; RF puede diferir en los últimos bits por el orden de sumas paralelas. No se redondearon métricas ni predicciones.
- Todos los archivos originales de datasets conservados. Se preservó la evidencia anterior por separado.
- Se actualizó el dashboard al target v2 y a la tabla común; el motor sigue configurado como Regresión Lineal. La interfaz no se volvió a certificar visualmente en esta ejecución; sí se verificó su sintaxis y se incluyen modelos/metadatos nuevos.

## Limitaciones que permanecen

1. **Población idéntica no significa protocolo idéntico.** ARIMA pronostica multi-step desde fin de TRAIN; ML usa historia observada semana a semana. Las diferencias no pueden atribuirse sólo al algoritmo. Para una comparación de algoritmos con igual información haría falta otro experimento con protocolo alineado; no se cambió el solicitado.
2. La intersección está condicionada por elegibilidad y éxito ARIMA. Sus resultados no se extrapolan automáticamente a las 156 series de test ML ni a todas las series de la red.
3. El objetivo es cantidad positiva observada en ERP, incluyendo ceros, no demanda neta, ventas en unidades equivalentes ni demanda perdida por faltantes. Al eliminar reversos cambia el fenómeno modelado.
4. Se rellenan huecos interiores como cero bajo continuidad de registro. No se expanden los extremos de las series depuradas. Las semanas extremas del dataset pueden ser parciales; no se interpretan como semanas completas confirmadas.
5. Las métricas son agregadas por observación y no ponderadas equitativamente por serie. La abundancia de ceros influye en su interpretación. No se presentaron intervalos ni pruebas de significancia.
6. Seleccionar un modelo después de ver el holdout es descriptivo y requiere validación futura independiente. No se tocaron hiperparámetros para favorecer a XGBoost.
7. Las predicciones negativas de regresión se mantienen, como en la evaluación; cualquier regla operativa de recorte debe evaluarse por separado. Esto es distinto de eliminar movimientos negativos del target.

## Archivos de código y notebooks modificados o nuevos

- `app.py`
- `test_comparacion_comun.py`
- `test_demanda_semanal.py`
- `notebooks/06_Entrenamiento_ARIMA.ipynb`
- `notebooks/07_Entrenamiento_MachineLearning.ipynb`
- `notebooks/08_EDA_Intermitencia_Series.ipynb`
- `notebooks/modelo_random_forest_semanal.json`
- `notebooks/modelo_regresion_lineal_semanal.json`
- `notebooks/modelo_xgboost_semanal.json`
- `scripts/validar_evidencia_semanal.py`
- `src/evaluacion_semanal.py`
- `src/experimento_semanal.py`
- `src/Modelos/ArimaPredictor.py`
- `src/Modelos/MachineLearningPipeline.py`
- `src/preprocessing/clean_data.py`
- `src/preprocessing/demanda_semanal.py`
- `src/preprocessing/prediccion_dashboard.py`

Además, se regeneró `resultados/semanal/` con el dataset analítico, auditoría, series semanales, particiones, predicciones, métricas propias y comunes, cobertura y manifiestos, y se exportaron los tres pipelines ML con JSON.
