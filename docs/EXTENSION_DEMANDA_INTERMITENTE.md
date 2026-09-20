# Extensión experimental para demanda altamente intermitente

## Objetivo y preservación

Se investiga si separar ocurrencia y magnitud mejora frente a la regresión directa. Se conserva el experimento base y la evidencia de intermitencia; ninguna salida de `resultados/semanal/`, artefacto previo, dataset RAW ni `app.py` se modificó. La extensión utiliza TRAIN ML=15170, TEST ML=3940 y la población científica fija de 3444 observaciones/129 series. No se eliminaron ceros ni se cambió el target.

## Métodos e inicialización

- Cero: siempre 0; naive: última cantidad semanal observada.
- Croston: suaviza por separado tamaño positivo e intervalo entre eventos y pronostica su cociente. SBA aplica el factor (1−alpha/2). Véase Syntetos y Boylan (2005), [The accuracy of intermittent demand estimates](https://www.sciencedirect.com/science/article/pii/S0169207004000792), y el desarrollo original de Croston (1972) allí referido.
- TSB: suaviza tamaño en eventos positivos y probabilidad de ocurrencia cada semana, incluso con ceros; pronostica tamaño por probabilidad. Teunter, Syntetos y Babai (2011), [Intermittent demand: Linking forecasting to inventory obsolescence](https://doi.org/10.1016/j.ejor.2011.05.018).
- Convención de arranque causal de esta implementación: pronóstico cero hasta observar la primera positiva; tamaño inicial=esa cantidad; intervalo inicial=número de semanas observadas hasta ella; probabilidad inicial=1/ese número. No se inspeccionan eventos futuros para inicializar. Antes de cada actualización se emite el pronóstico de la semana objetivo. TSB usa alpha para tamaño y beta para probabilidad.
- Two-stage: tres clasificadores (Logistic Regression, RF, XGBoost) y tres regresores positivos (RL, RF, XGBoost). Se comparan las etapas por separado en validación, no las nueve combinaciones sobre TEST. El pronóstico es magnitud si P>=umbral, y 0 en otro caso. La magnitud se restringe a max(0,predicción) como regla estructural fijada antes de CV; se aplica igual a todos los regresores y no altera targets ni predicciones del experimento base.

## Features y ausencia de futuro

Se conservan lag_1/2/4/8, rolling_mean_4/8 con shift(1), mes, semana_del_anio y trimestre, junto con identificadores sucursal/producto conocidos. El mes/trimestre corresponde al domingo W-SUN aunque la semana cruce un mes. No hay día de semana, fin de semana, total transaccional ni unidades contemporáneas.

Se añaden ocho descriptores: distancia al último evento positivo, conteos/tasas positivos en 4/8 semanas previas, media histórica positiva y dos últimas magnitudes positivas. Todas se calculan antes de actualizar con la cantidad objetivo. Sin eventos previos, magnitudes/media=0 y distancia=semanas observadas+1 (distancia censurada desde el inicio, no tiempo real desde una venta desconocida). El calendario conserva la semana cruzada como historia disponible; se excluye de scoring, no se comprimen lags.

El reporte `auditoria_leakage.csv` perturba el target actual y todo el futuro en dos fechas y comprueba que ninguna feature del presente/pasado cambie. Las pruebas también verifican estados Croston/TSB, rolling desplazado, selección sin TEST y conservación de artefactos.

## Selección exclusivamente dentro de TRAIN

TimeSeriesSplit(n_splits=3) sobre semanas únicas, con ventanas expansivas. Cada fold entrena sólo antes de su validación. Features/estados usan la historia observada anterior dentro de validación: protocolo secuencial de una semana; parámetros ML fijos por fold. One-Hot se ajusta dentro del fold; Logistic Regression escala numéricas dentro de TRAIN. Se fijaron C=1 y max_iter=2000; RF=200 árboles/profundidad10/hoja2; XGB=100 árboles/profundidad3/learning_rate0.05; semilla42. No se ampliaron rejillas tras ver TEST.

Criterios fijados: clasificador con menor Brier OOF; magnitud con menor MSE OOF exclusivamente en semanas positivas; umbral con menor MSE del pronóstico final sobre **todas** las validaciones, incluidos ceros. Se prueban umbrales [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; 0.5 es sólo un candidato. Empates de umbral eligen el menor; empates entre estimadores siguen el orden declarado. Se comparan features básicas y enriquecidas mediante MSE final OOF. No se elige por métricas TEST.

Croston/SBA prueban alpha en [0.05, 0.1, 0.2]; TSB combina esos alpha/beta (9 opciones), seleccionados globalmente por MSE OOF, sin tuning por serie en TEST. Cero y naive carecen de parámetros.

Configuración congelada antes de inferencia TEST:

```json
{
  "features": "basicas",
  "clasificador": "XGBoost",
  "regresor": "XGBoost",
  "umbral": 0.5,
  "MSE_cv": 0.3014259953690401,
  "MAE_cv": 0.14988424914162404
}
```

Parámetros intermitentes:

```json
{
  "Cero": {},
  "Naive_ultimo": {},
  "Croston": {
    "alpha": 0.2
  },
  "SBA": {
    "alpha": 0.2
  },
  "TSB": {
    "alpha": 0.2,
    "beta": 0.1
  }
}
```

Ablación de features, exclusivamente CV:

| features | clasificador | regresor | umbral | MSE_cv | MAE_cv |
| --- | --- | --- | --- | --- | --- |
| basicas | XGBoost | XGBoost | 0.5000 | 0.3014 | 0.1499 |
| intermitentes | XGBoost | Random Forest | 0.5000 | 0.3132 | 0.1510 |

Media y desviación muestral entre folds (es diagnóstico interno de selección, no estimación independiente ni intervalo de confianza):

| Modelo | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std | MASE_mean | MASE_std | RMSSE_mean | RMSSE_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cero | 0.1367 | 0.0318 | 0.5558 | 0.0879 | -0.0637 | 0.0097 | 0.6187 | 0.2057 | 0.6397 | 0.1405 |
| Croston | 0.2271 | 0.0078 | 0.4971 | 0.0315 | 0.1297 | 0.1427 | 2.3650 | 0.2513 | 0.8808 | 0.1325 |
| Naive_ultimo | 0.1914 | 0.0307 | 0.6448 | 0.0709 | -0.4463 | 0.1174 | 1.0505 | 0.3369 | 0.8330 | 0.1665 |
| SBA | 0.2154 | 0.0099 | 0.4884 | 0.0364 | 0.1626 | 0.1204 | 2.1871 | 0.2451 | 0.8481 | 0.1330 |
| TSB | 0.1706 | 0.0143 | 0.4680 | 0.0371 | 0.2315 | 0.1106 | 0.9497 | 0.2150 | 0.6204 | 0.1076 |
| TwoStage_basicas | 0.1496 | 0.0069 | 0.5483 | 0.0299 | -0.0656 | 0.2187 | 0.6408 | 0.1705 | 0.6436 | 0.1215 |
| TwoStage_intermitentes | 0.1510 | 0.0043 | 0.5599 | 0.0223 | -0.1223 | 0.2857 | 0.6432 | 0.1623 | 0.6427 | 0.1173 |

Las mismas validaciones se utilizan para seleccionar etapas, umbral y features; no es CV anidada. Puede haber optimismo de selección. TEST se evaluó una vez para las configuraciones seleccionadas y no se usó para revisarlas. Aunque se congela esta extensión antes de inferencia, el holdout ya fue observado en trabajos anteriores: no se presenta como prueba nueva nunca vista.

## Métricas y escalado

MAE/RMSE/R² globales se calculan concatenando observaciones comunes. R² se marca no definido cuando el target es constante o n<2; no se sustituye por 0/1. No se calcula MAPE.

MASE_i=MAE_i / mean(|diff(TRAIN_i)|); RMSSE_i=sqrt(MSE_i / mean(diff(TRAIN_i)²)), con rezago no estacional 1. Se reporta la media no ponderada de estos valores por serie (macro). Los denominadores usan sólo el calendario TRAIN disponible, incluidos sus ceros; en cada fold se recalculan hasta su origen. No usan TEST. Referencia de errores escalados: Hyndman y Koehler (2006), [Another look at measures of forecast accuracy](https://fpp.robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/), y [Forecasting: Principles and Practice, evaluación de precisión](https://otexts.com/fpp3/accuracy.html).

Con escala cero o sin historia suficiente, MASE/RMSSE no están definidos. No se agrega epsilon ni se elimina esa serie de MAE/RMSE/R². En la comparación común se reportan **92 series y 2418 observaciones** con escala válida; las restantes siguen en los 3444 casos. MASE/RMSSE y métricas globales tienen ponderaciones/poblaciones distintas y no deben confundirse.

## Resultados globales sobre población común

| Modelo | MAE | RMSE | R2 | MASE | RMSSE | MAE_positivas | RMSE_positivas | porcentaje_predicciones_negativas | observaciones | series |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | 0.2306 | 0.5396 | 0.3219 | 1.4504 | 0.6757 | 0.9787 | 1.3638 | 19.8606 | 3444 | 129 |
| Random Forest | 0.2270 | 0.5487 | 0.2989 | 1.1656 | 0.6471 | 0.9794 | 1.3479 | 0.0000 | 3444 | 129 |
| XGBoost | 0.2252 | 0.5489 | 0.2984 | 1.3513 | 0.6593 | 1.0006 | 1.3692 | 0.0000 | 3444 | 129 |
| ARIMA (Optimizado ADF/AIC) | 0.2363 | 0.5797 | 0.2175 | 1.4249 | 0.7470 | 1.0306 | 1.3931 | 30.4297 | 3444 | 129 |
| TwoStage | 0.2052 | 0.6101 | 0.1331 | 0.7550 | 0.6449 | 1.1946 | 1.4307 | 0.0000 | 3444 | 129 |
| Cero | 0.2009 | 0.6854 | -0.0940 | 0.7477 | 0.6646 | 1.6282 | 1.9512 | 0.0000 | 3444 | 129 |
| Naive_ultimo | 0.2427 | 0.7225 | -0.2157 | 1.2217 | 0.8264 | 1.2659 | 1.6789 | 0.0000 | 3444 | 129 |
| Croston | 0.2470 | 0.5626 | 0.2629 | 2.7292 | 0.8340 | 1.0017 | 1.3606 | 0.0000 | 3444 | 129 |
| SBA | 0.2393 | 0.5601 | 0.2694 | 2.5277 | 0.8108 | 1.0386 | 1.4015 | 0.0000 | 3444 | 129 |
| TSB | 0.2126 | 0.5453 | 0.3076 | 1.0697 | 0.6249 | 1.0001 | 1.3656 | 0.0000 | 3444 | 129 |

En esta población, menor MAE: **Cero**; menor RMSE: **Regresión Lineal**. No se interpreta un mínimo aislado como superioridad universal.

## Semanas cero y positivas

| Modelo | Tipo_demanda | MAE | RMSE | R2 | observaciones | series |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | cero | 0.1253 | 0.2653 | — | 3019 | 129 |
| Regresión Lineal | positiva | 0.9787 | 1.3638 | -0.6090 | 425 | 66 |
| Random Forest | cero | 0.1211 | 0.2961 | — | 3019 | 129 |
| Random Forest | positiva | 0.9794 | 1.3479 | -0.5717 | 425 | 66 |
| XGBoost | cero | 0.1160 | 0.2824 | — | 3019 | 129 |
| XGBoost | positiva | 1.0006 | 1.3692 | -0.6219 | 425 | 66 |
| ARIMA (Optimizado ADF/AIC) | cero | 0.1245 | 0.3318 | — | 3019 | 129 |
| ARIMA (Optimizado ADF/AIC) | positiva | 1.0306 | 1.3931 | -0.6791 | 425 | 66 |
| TwoStage | cero | 0.0660 | 0.3695 | — | 3019 | 129 |
| TwoStage | positiva | 1.1946 | 1.4307 | -0.7709 | 425 | 66 |
| Cero | cero | 0.0000 | 0.0000 | — | 3019 | 129 |
| Cero | positiva | 1.6282 | 1.9512 | -2.2936 | 425 | 66 |
| Naive_ultimo | cero | 0.0987 | 0.4458 | — | 3019 | 129 |
| Naive_ultimo | positiva | 1.2659 | 1.6789 | -1.4386 | 425 | 66 |
| Croston | cero | 0.1408 | 0.3170 | — | 3019 | 129 |
| Croston | positiva | 1.0017 | 1.3606 | -0.6015 | 425 | 66 |
| SBA | cero | 0.1267 | 0.2853 | — | 3019 | 129 |
| SBA | positiva | 1.0386 | 1.4015 | -0.6993 | 425 | 66 |
| TSB | cero | 0.1017 | 0.2768 | — | 3019 | 129 |
| TSB | positiva | 1.0001 | 1.3656 | -0.6133 | 425 | 66 |

## Patrones intermitente y lumpy

Las etiquetas son las ya calculadas exclusivamente en TRAIN; no se vuelven a elegir según esta extensión.

| Modelo | clasificacion_demanda | MAE | RMSE | R2 | observaciones | series |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | intermitente | 0.3839 | 0.7201 | 0.2814 | 1669 | 63 |
| Regresión Lineal | lumpy | 0.4955 | 0.7735 | 0.2745 | 104 | 5 |
| Random Forest | intermitente | 0.3852 | 0.7352 | 0.2510 | 1669 | 63 |
| Random Forest | lumpy | 0.4589 | 0.7318 | 0.3505 | 104 | 5 |
| XGBoost | intermitente | 0.3727 | 0.7337 | 0.2540 | 1669 | 63 |
| XGBoost | lumpy | 0.4646 | 0.7686 | 0.2836 | 104 | 5 |
| ARIMA (Optimizado ADF/AIC) | intermitente | 0.4172 | 0.7809 | 0.1550 | 1669 | 63 |
| ARIMA (Optimizado ADF/AIC) | lumpy | 0.5415 | 0.7703 | 0.2804 | 104 | 5 |
| TwoStage | intermitente | 0.3592 | 0.8086 | 0.0938 | 1669 | 63 |
| TwoStage | lumpy | 0.4390 | 0.8981 | 0.0218 | 104 | 5 |
| Cero | intermitente | 0.3553 | 0.9208 | -0.1750 | 1669 | 63 |
| Cero | lumpy | 0.4519 | 1.0143 | -0.2477 | 104 | 5 |
| Naive_ultimo | intermitente | 0.4218 | 0.9600 | -0.2771 | 1669 | 63 |
| Naive_ultimo | lumpy | 0.5385 | 1.0470 | -0.3293 | 104 | 5 |
| Croston | intermitente | 0.4034 | 0.7384 | 0.2445 | 1669 | 63 |
| Croston | lumpy | 0.5346 | 0.8121 | 0.2002 | 104 | 5 |
| SBA | intermitente | 0.3929 | 0.7382 | 0.2449 | 1669 | 63 |
| SBA | lumpy | 0.5186 | 0.8034 | 0.2172 | 104 | 5 |
| TSB | intermitente | 0.3692 | 0.7271 | 0.2672 | 1669 | 63 |
| TSB | lumpy | 0.5559 | 0.8044 | 0.2152 | 104 | 5 |

Todos los restantes patrones y excepciones permanecen en el CSV. La cobertura común y los ceros se conservan exactamente.

## Mejora relativa y respuesta a la pregunta de investigación

La separación con puerta dura no mostró una mejora simultánea frente a RL en MAE/RMSE global y en semanas positivas. No se declara superioridad del enfoque.

Respecto de RL, TwoStage cambia MAE en una mejora relativa de **10.9899%**, RMSE **-13.0648%**, MAE positivo **-22.0700%** y RMSE positivo **-4.9084%**. Valores negativos significan deterioro. La tabla completa compara cada modelo con cero, RL (referencia previa de menor RMSE entre ML) y XGBoost (referencia previa de menor MAE entre ML). No se usa porcentaje relativo de R²; se informa diferencia absoluta.

| Modelo | Referencia | Mejora_pct_MAE | Mejora_pct_RMSE | Mejora_pct_MASE | Mejora_pct_RMSSE | Mejora_pct_MAE_positivas | Mejora_pct_RMSE_positivas | Cambio_absoluto_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TwoStage | Cero | -2.1500 | 10.9820 | -0.9753 | 2.9583 | 26.6292 | 26.6734 | 0.2271 |
| TwoStage | Regresión Lineal | 10.9899 | -13.0648 | 47.9497 | 4.5530 | -22.0700 | -4.9084 | -0.1888 |
| TwoStage | XGBoost | 8.8529 | -11.1589 | 44.1314 | 2.1732 | -19.3874 | -4.4912 | -0.1653 |

TSB reduce MAE un 5.6057% y RMSE un 0.6612% respecto a XGBoost; frente a RL reduce MAE un 7.8188%, pero aumenta RMSE un 1.0421%. Es una mejora descriptiva respecto a XGBoost, no evidencia de superioridad estadística ni de mejora frente a todos los modelos. El pronóstico cero alcanza el menor MAE global, pero presenta R² negativo y mayores errores en semanas positivas.

La selección temporal prefirió las variables básicas: añadir las ocho variables de intermitencia aumentó el MSE de validación de 0.301426 a 0.313195. El umbral final 0.5 fue elegido entre doce candidatos mediante TRAIN/CV; no se impuso automáticamente.

La respuesta se limita a la puerta dura, estimadores, features, rejillas y protocolo probados. Un resultado desfavorable no demuestra un límite predictivo fundamental de los datos ni descarta otras variantes de dos etapas. Tampoco autoriza ajustar el umbral con TEST. La regla de puerta dura no equivale a la esperanza p·magnitud; esa variante no se evaluó en este experimento solicitado.

## Cobertura y límites

| Modelo | Observaciones_propias | Series_propias | Observaciones_comunes | Series_comunes |
| --- | --- | --- | --- | --- |
| TwoStage | 3940 | 156 | 3444 | 129 |
| Cero | 3940 | 156 | 3444 | 129 |
| Naive_ultimo | 3940 | 156 | 3444 | 129 |
| Croston | 3940 | 156 | 3444 | 129 |
| SBA | 3940 | 156 | 3444 | 129 |
| TSB | 3940 | 156 | 3444 | 129 |
| Regresión Lineal | 3940 | 156 | 3444 | 129 |
| Random Forest | 3940 | 156 | 3444 | 129 |
| XGBoost | 3940 | 156 | 3444 | 129 |
| ARIMA (Optimizado ADF/AIC) | 3444 | 129 | 3444 | 129 |

Los nuevos métodos cubren las 3940 claves ML, pero se comparan científicamente sobre las mismas 3444 claves de los cuatro modelos originales. Los resultados propios de 3940 están separados en `metricas_test.csv`; no se comparan directamente con ARIMA de cobertura menor.

ARIMA sigue siendo fijo multi-step; ML y los métodos nuevos usan historia semanal observada y horizonte de una semana. La igualdad de población no elimina esa diferencia de información. La validación de mejora entre ML y nuevos métodos es más alineada en protocolo que frente a ARIMA. Persisten abundancia de ceros, magnitudes ERP sin conversión de unidades, posibles semanas extremas parciales y ausencia de demanda perdida por stock-outs. Las cinco series lumpy son una muestra pequeña. No se estimó significancia estadística ni se midió beneficio económico. El RMSE de folds presenta variabilidad temporal y las comparaciones son descriptivas.

## Reproducción y archivos

`python scripts/ejecutar_extension_intermitente.py` realiza la extensión **sólo si no existe su evaluación TEST**. Si ya está materializada, se bloquea el reentrenamiento/repronóstico accidental. Para verificar: `python scripts/validar_extension_intermitente.py`. Para regenerar exclusivamente tablas e informe desde predicciones bloqueadas: `python scripts/resumir_extension_intermitente.py`. No cambia configuración ni ejecuta fit/predict.

Se crean únicamente módulos/scripts/tests nuevos, `docs/EXTENSION_DEMANDA_INTERMITENTE.md`, el registro de integridad y `resultados/extension_intermitente/`. Contiene todos los CSV solicitados, predicciones por clave, OOF, candidatos CV, selección congelada, escalas TRAIN, cobertura, mejoras relativas, pipeline nuevo y manifiesto. No se modifica Streamlit ni la Discusión/Conclusiones existentes. No se hace merge a main.
