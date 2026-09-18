# Análisis de intermitencia y evidencia para despliegue

Este análisis adicional usa exclusivamente las predicciones comunes existentes. No se entrenaron modelos, no se alteraron hiperparámetros, target, prueba ni predicciones y no se modificó `app.py`. Los borradores del final no sustituyen la Discusión ni las Conclusiones existentes del artículo.

## A. Metodología y referencia

El EDA anterior sólo contaba semanas cero y demanda promedio; sus tablas completa/evaluable no definían categorías ADI/CV². Se conserva esa evidencia. La clasificación nueva usa sólo semanas enteramente anteriores a 2026-01-01: `semana + 1 día <= corte`. No consulta valores del test y mantiene la historia anterior a la eliminación de lags. La semana cruzada no participa en la clasificación.

Se adopta la separación descriptiva ADI=1.32 y CV²=0.49 de Syntetos, Boylan y Croston (2005), *On the categorization of demand patterns*, Journal of the Operational Research Society, 56, 495–503, [DOI](https://doi.org/10.1057/palgrave.jors.2601841), figura 3. El trabajo fundamenta estos cortes para otros métodos de pronóstico; aquí sirven como taxonomía, no como garantía de superioridad para RL/RF/XGBoost/ARIMA.

Estimadores explícitos: ADI=N/N+ (semanas observadas / semanas positivas, incluyendo ceros iniciales/finales del calendario disponible); CV²=(s+/media+)², con s+ muestral (`ddof=1`) calculada únicamente sobre tamaños positivos. Esta elección de estimadores, tratamiento de límites y casos insuficientes es una convención reproducible del presente análisis; no se atribuye al artículo una regla para casos no estimables.

| Patrón | ADI | CV² |
|---|---|---|
| Suave | <1.32 | <0.49 |
| Errática | <1.32 | >=0.49 |
| Intermitente | >=1.32 | <0.49 |
| Lumpy | >=1.32 | >=0.49 |

La igualdad se asigna al lado superior. `grupo_demanda` usa solamente ADI: menor intermitencia (<1.32), mayor intermitencia (>=1.32). No se divide por desempeño. Con una positiva se estima ADI y el grupo, pero CV² no es estimable y se etiqueta `positivas_insuficientes`. Con ninguna positiva, ADI/CV² quedan ausentes y se conserva el grupo separado `sin_demanda_positiva`. Sin historia TRAIN se marca `sin_historia_train`. No se elimina ninguna observación por estas excepciones.

## B. Distribución y posibilidad real de contraste

| clasificacion_demanda | Series_comunes | Series_dataset |
| --- | --- | --- |
| suave | 1 | 1 |
| erratica | 0 | 0 |
| intermitente | 63 | 100 |
| lumpy | 5 | 6 |
| positivas_insuficientes | 23 | 39 |
| sin_demanda_positiva | 37 | 72 |
| sin_historia_train | 0 | 22 |

En la población común hay 1 serie suave, 0 erráticas, 63 intermitentes, 5 lumpy, 23 con una sola positiva y 37 sin positivas en TRAIN. La comparación binaria incluye 1 serie de menor intermitencia y 91 de mayor intermitencia; las 37 sin positivas permanecen aparte. Por tanto, no hay evidencia suficiente para generalizar diferencias entre demanda continua y demanda intermitente: el primer grupo tiene sólo una serie (30 observaciones de TEST). Las 5 lumpy aportan 104 observaciones.

## C. Ceros y estabilidad de las etiquetas

En TRAIN de las series comunes hay 11098 semanas, 1225 positivas y 88.9620% ceros (ponderado por semanas). El promedio de los porcentajes por serie es 89.2987%, y la mediana es 97.1154%. Son denominadores distintos y no deben intercambiarse.

En TEST común: **3019 ceros de 3444 (87.6597%)** y **425 positivas (12.3403%)**. Existen 63 series con sólo ceros en TEST y 66 con alguna positiva. Una etiqueta sin positivas en TRAIN no asegura ausencia de demanda futura: ese grupo contiene 7 semanas positivas en TEST.

Los intervalos de ceros derivan de cuartiles del porcentaje TRAIN entre las 129 series comunes, con cada serie ponderada una vez. Se colapsan los cortes duplicados, obteniendo tres intervalos; el primero incluye ambos extremos y los restantes son abiertos a izquierda/cerrados a derecha. No se usan errores ni métricas para definirlos.

| intervalo_ceros | limite_inferior | limite_superior | inferior_inclusivo |
| --- | --- | --- | --- |
| Q1 | 18.1818 | 88.4615 | True |
| Q2 | 88.4615 | 97.1154 | False |
| Q3 | 97.1154 | 100.0000 | False |

| Modelo | intervalo_ceros | MAE | RMSE | R2 | Observaciones | Series |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | Q1 | 0.6313 | 0.9636 | 0.2048 | 900 | 33 |
| Random Forest | Q1 | 0.6435 | 0.9815 | 0.1750 | 900 | 33 |
| XGBoost | Q1 | 0.6259 | 0.9847 | 0.1697 | 900 | 33 |
| ARIMA (Optimizado ADF/AIC) | Q1 | 0.6375 | 0.9713 | 0.1921 | 900 | 33 |
| Regresión Lineal | Q2 | 0.1841 | 0.3974 | 0.0468 | 817 | 32 |
| Random Forest | Q2 | 0.1781 | 0.4020 | 0.0245 | 817 | 32 |
| XGBoost | Q2 | 0.1657 | 0.3963 | 0.0520 | 817 | 32 |
| ARIMA (Optimizado ADF/AIC) | Q2 | 0.2547 | 0.5775 | -1.0131 | 817 | 32 |
| Regresión Lineal | Q3 | 0.0438 | 0.1488 | -0.0701 | 1727 | 64 |
| Random Forest | Q3 | 0.0332 | 0.1479 | -0.0583 | 1727 | 64 |
| XGBoost | Q3 | 0.0445 | 0.1458 | -0.0277 | 1727 | 64 |
| ARIMA (Optimizado ADF/AIC) | Q3 | 0.0186 | 0.1439 | -0.0013 | 1727 | 64 |

| Modelo | Metrica | Spearman_por_serie | Series |
| --- | --- | --- | --- |
| Regresión Lineal | MAE | -0.8319 | 129 |
| Regresión Lineal | RMSE | -0.7759 | 129 |
| Random Forest | MAE | -0.8182 | 129 |
| Random Forest | RMSE | -0.7998 | 129 |
| XGBoost | MAE | -0.8099 | 129 |
| XGBoost | RMSE | -0.7778 | 129 |
| ARIMA (Optimizado ADF/AIC) | MAE | -0.9089 | 129 |
| ARIMA (Optimizado ADF/AIC) | RMSE | -0.8499 | 129 |

Las correlaciones son descriptivas entre porcentaje de ceros TRAIN y error TEST por serie. No se calculan p-valores ni se afirma causalidad; escala de demanda, producto, sucursal, longitud de serie y régimen temporal pueden influir conjuntamente.

## D. Métricas por patrón y grupo

| Modelo | Clasificacion_demanda | MAE | RMSE | R2 | Observaciones | Series |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | suave | 0.8246 | 1.0476 | -0.0725 | 30 | 1 |
| Random Forest | suave | 0.8666 | 1.1091 | -0.2021 | 30 | 1 |
| XGBoost | suave | 0.9025 | 1.1102 | -0.2043 | 30 | 1 |
| ARIMA (Optimizado ADF/AIC) | suave | 0.8666 | 1.1145 | -0.2137 | 30 | 1 |
| Regresión Lineal | erratica | — | — | — | 0 | 0 |
| Random Forest | erratica | — | — | — | 0 | 0 |
| XGBoost | erratica | — | — | — | 0 | 0 |
| ARIMA (Optimizado ADF/AIC) | erratica | — | — | — | 0 | 0 |
| Regresión Lineal | intermitente | 0.3839 | 0.7201 | 0.2814 | 1669 | 63 |
| Random Forest | intermitente | 0.3852 | 0.7352 | 0.2510 | 1669 | 63 |
| XGBoost | intermitente | 0.3727 | 0.7337 | 0.2540 | 1669 | 63 |
| ARIMA (Optimizado ADF/AIC) | intermitente | 0.4172 | 0.7809 | 0.1550 | 1669 | 63 |
| Regresión Lineal | lumpy | 0.4955 | 0.7735 | 0.2745 | 104 | 5 |
| Random Forest | lumpy | 0.4589 | 0.7318 | 0.3505 | 104 | 5 |
| XGBoost | lumpy | 0.4646 | 0.7686 | 0.2836 | 104 | 5 |
| ARIMA (Optimizado ADF/AIC) | lumpy | 0.5415 | 0.7703 | 0.2804 | 104 | 5 |
| Regresión Lineal | positivas_insuficientes | 0.0681 | 0.2249 | -0.0191 | 615 | 23 |
| Random Forest | positivas_insuficientes | 0.0537 | 0.2229 | -0.0008 | 615 | 23 |
| XGBoost | positivas_insuficientes | 0.0641 | 0.2217 | 0.0100 | 615 | 23 |
| ARIMA (Optimizado ADF/AIC) | positivas_insuficientes | 0.0443 | 0.2234 | -0.0051 | 615 | 23 |
| Regresión Lineal | sin_demanda_positiva | 0.0343 | 0.1047 | -0.1319 | 1026 | 37 |
| Random Forest | sin_demanda_positiva | 0.0315 | 0.1067 | -0.1764 | 1026 | 37 |
| XGBoost | sin_demanda_positiva | 0.0377 | 0.1016 | -0.0659 | 1026 | 37 |
| ARIMA (Optimizado ADF/AIC) | sin_demanda_positiva | 0.0078 | 0.0987 | -0.0063 | 1026 | 37 |

| Modelo | grupo_demanda | MAE | RMSE | R2 | Observaciones | Series |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | menor_intermitencia | 0.8246 | 1.0476 | -0.0725 | 30 | 1 |
| Random Forest | menor_intermitencia | 0.8666 | 1.1091 | -0.2021 | 30 | 1 |
| XGBoost | menor_intermitencia | 0.9025 | 1.1102 | -0.2043 | 30 | 1 |
| ARIMA (Optimizado ADF/AIC) | menor_intermitencia | 0.8666 | 1.1145 | -0.2137 | 30 | 1 |
| Regresión Lineal | mayor_intermitencia | 0.3075 | 0.6336 | 0.3013 | 2388 | 91 |
| Random Forest | mayor_intermitencia | 0.3030 | 0.6433 | 0.2798 | 2388 | 91 |
| XGBoost | mayor_intermitencia | 0.2972 | 0.6439 | 0.2785 | 2388 | 91 |
| ARIMA (Optimizado ADF/AIC) | mayor_intermitencia | 0.3266 | 0.6818 | 0.1911 | 2388 | 91 |
| Regresión Lineal | sin_demanda_positiva | 0.0343 | 0.1047 | -0.1319 | 1026 | 37 |
| Random Forest | sin_demanda_positiva | 0.0315 | 0.1067 | -0.1764 | 1026 | 37 |
| XGBoost | sin_demanda_positiva | 0.0377 | 0.1016 | -0.0659 | 1026 | 37 |
| ARIMA (Optimizado ADF/AIC) | sin_demanda_positiva | 0.0078 | 0.0987 | -0.0063 | 1026 | 37 |

Se observó que XGBoost presenta menor MAE en las 63 series intermitentes y RL menor RMSE en ese patrón; RF presenta menor MAE/RMSE en las 5 lumpy. El grupo suave favorece RL en ambas métricas, pero contiene una única serie. No se generaliza ese resultado a demanda continua en toda la red.

R² se conserva incluso cuando es negativo. Se informa `Varianza_y_real`, SST, porcentaje de ceros, estado y advertencias. Las alertas señalan varianza inferior a la global o ceros superiores al porcentaje global; son comparaciones descriptivas, no umbrales de inestabilidad universal. Los grupos con una sola positiva o ninguna positiva en TRAIN muestran R² próximo a cero/negativo y baja varianza TEST. Cuando SST=0 o n<2, R² no está definido y se escribe vacío/NaN con motivo, nunca un 0 o 1 artificial. La categoría errática se muestra sin datos, no con error cero.

## E. Errores sobre semanas con demanda cero

| Modelo | MAE | RMSE | Observaciones | Series | R2_estado |
| --- | --- | --- | --- | --- | --- |
| Regresión Lineal | 0.1253 | 0.2653 | 3019 | 129 | no_definido_target_constante |
| Random Forest | 0.1211 | 0.2961 | 3019 | 129 | no_definido_target_constante |
| XGBoost | 0.1160 | 0.2824 | 3019 | 129 | no_definido_target_constante |
| ARIMA (Optimizado ADF/AIC) | 0.1245 | 0.3318 | 3019 | 129 | no_definido_target_constante |

R² no se usa aquí porque y_real es constante. XGBoost obtiene el menor MAE; RL obtiene el menor RMSE en este subconjunto.

## F. Errores sobre semanas con demanda positiva

| Modelo | MAE | RMSE | R2 | Observaciones | Series |
| --- | --- | --- | --- | --- | --- |
| Regresión Lineal | 0.9787 | 1.3638 | -0.6090 | 425 | 66 |
| Random Forest | 0.9794 | 1.3479 | -0.5717 | 425 | 66 |
| XGBoost | 1.0006 | 1.3692 | -0.6219 | 425 | 66 |
| ARIMA (Optimizado ADF/AIC) | 1.0306 | 1.3931 | -0.6791 | 425 | 66 |

En 425 semanas positivas de 66 series, RL presenta el menor MAE (0.9787), muy cercano al de RF (0.9794); RF presenta el menor RMSE (1.3479). XGBoost obtiene MAE 1.0006 y RMSE 1.3692. Todos los R² condicionados a positivas son negativos: respecto de la media de este subconjunto como referencia descriptiva, los errores cuadrados son mayores. Esa media no es un pronosticador disponible a priori y este análisis condicionado al resultado no sustituye una evaluación prospectiva.

## G. Predicciones negativas sin recorte

| Modelo | Predicciones_negativas | Porcentaje_predicciones_negativas | Minimo_predicho | MAE_negativas | RMSE_negativas | Negativas_con_y_cero | Negativas_con_y_positivo | Mediana_absoluta_negativas | Percentil95_absoluto_negativas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | 684 | 19.8606 | -0.0552 | 0.0346 | 0.1573 | 676 | 8 | 0.0191 | 0.0416 |
| Random Forest | 0 | 0.0000 | 0.0068 | — | — | 0 | 0 | — | — |
| XGBoost | 0 | 0.0000 | 0.0237 | — | — | 0 | 0 | — | — |
| ARIMA (Optimizado ADF/AIC) | 1048 | 30.4297 | -0.2625 | 0.0119 | 0.1244 | 1038 | 10 | 0.0000 | 0.0000 |

El criterio es estrictamente y_pred<0, sin tolerancia ni clip. RL tiene 684 negativas (676 con y_real=0, 8 con y_real>0), mínimo −0.055221. ARIMA tiene 1048 (1038/10), mínimo −0.262526; la mediana y percentil 95 de su magnitud son aproximadamente 0.000005, de modo que el conteo de signos por sí solo oculta magnitudes muy distintas. RF y XGBoost no presentan negativas en esta evaluación; ello no constituye una garantía universal para cualquier dato o versión. Un eventual recorte requiere una evaluación separada y no se aplicó.

## H. Desempeño por serie y cobertura

Mínimos entre los cuatro modelos:

| Modelo | Metrica | Series_minimo_exclusivo | Series_minimo_compartido | Series_minimo_incluyendo_empates | Series_totales |
| --- | --- | --- | --- | --- | --- |
| Regresión Lineal | MAE | 8 | 0 | 8 | 129 |
| Random Forest | MAE | 34 | 0 | 34 | 129 |
| XGBoost | MAE | 14 | 0 | 14 | 129 |
| ARIMA (Optimizado ADF/AIC) | MAE | 73 | 0 | 73 | 129 |
| Regresión Lineal | RMSE | 10 | 0 | 10 | 129 |
| Random Forest | RMSE | 36 | 0 | 36 | 129 |
| XGBoost | RMSE | 16 | 0 | 16 | 129 |
| ARIMA (Optimizado ADF/AIC) | RMSE | 67 | 0 | 67 | 129 |

Se reconocen empates con `rtol=1e-9, atol=1e-12`; se conservan las métricas originales. No hubo empates en estos conteos. El número de mínimos da el mismo peso a series con magnitudes y longitudes distintas; no equivale al MAE/RMSE agregado por observación ni implica superioridad global.

| Modelo | Metrica | Series_minimo_exclusivo | Series_minimo_compartido | Series_minimo_incluyendo_empates | Series_totales | Tipo_serie_test |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | MAE | 2 | 0 | 2 | 63 | sólo_ceros |
| Random Forest | MAE | 23 | 0 | 23 | 63 | sólo_ceros |
| XGBoost | MAE | 2 | 0 | 2 | 63 | sólo_ceros |
| ARIMA (Optimizado ADF/AIC) | MAE | 36 | 0 | 36 | 63 | sólo_ceros |
| Regresión Lineal | RMSE | 1 | 0 | 1 | 63 | sólo_ceros |
| Random Forest | RMSE | 21 | 0 | 21 | 63 | sólo_ceros |
| XGBoost | RMSE | 5 | 0 | 5 | 63 | sólo_ceros |
| ARIMA (Optimizado ADF/AIC) | RMSE | 36 | 0 | 36 | 63 | sólo_ceros |
| Regresión Lineal | MAE | 6 | 0 | 6 | 66 | alguna_demanda_positiva |
| Random Forest | MAE | 11 | 0 | 11 | 66 | alguna_demanda_positiva |
| XGBoost | MAE | 12 | 0 | 12 | 66 | alguna_demanda_positiva |
| ARIMA (Optimizado ADF/AIC) | MAE | 37 | 0 | 37 | 66 | alguna_demanda_positiva |
| Regresión Lineal | RMSE | 9 | 0 | 9 | 66 | alguna_demanda_positiva |
| Random Forest | RMSE | 15 | 0 | 15 | 66 | alguna_demanda_positiva |
| XGBoost | RMSE | 11 | 0 | 11 | 66 | alguna_demanda_positiva |
| ARIMA (Optimizado ADF/AIC) | RMSE | 31 | 0 | 31 | 66 | alguna_demanda_positiva |

La separación entre series con sólo ceros y con alguna positiva evita interpretar automáticamente muchos mínimos de ARIMA como ventaja sobre semanas de demanda efectiva.

Cobertura propia, **sin mezclar métricas**:

| Modelo | Observaciones_evaluadas | Series_evaluadas | Observaciones_comunes | Series_comunes | Observaciones_fuera_comun |
| --- | --- | --- | --- | --- | --- |
| Regresión Lineal | 3940 | 156 | 3444 | 129 | 496 |
| Random Forest | 3940 | 156 | 3444 | 129 | 496 |
| XGBoost | 3940 | 156 | 3444 | 129 | 496 |
| ARIMA (Optimizado ADF/AIC) | 3444 | 129 | 3444 | 129 | 0 |

En un sistema multisucursal, los tres ML evaluaron 3940 observaciones de 156 series y ARIMA 3444 de 129. Las 496 observaciones/27 series adicionales de ML indican aplicabilidad histórica bajo los filtros vigentes, no precisión adicional demostrada ni garantía de cobertura futura. Todas las métricas del presente informe usan sólo la población común. ARIMA permanece como comparación científica, no como motor preseleccionado del dashboard.

## I. Comparación de candidatos operativos

| Modelo | MAE_global | RMSE_global | R2_global | MAE_menor_intermitencia | RMSE_menor_intermitencia | Series_menor_intermitencia | MAE_mayor_intermitencia | RMSE_mayor_intermitencia | Series_mayor_intermitencia | MAE_demanda_positiva | RMSE_demanda_positiva | Predicciones_negativas | Porcentaje_predicciones_negativas | Series_cubiertas | Observaciones_cubiertas | Observaciones_metricas_comunes | Series_metricas_comunes | Criterio_cobertura |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regresión Lineal | 0.2306 | 0.5396 | 0.3219 | 0.8246 | 1.0476 | 1 | 0.3075 | 0.6336 | 91 | 0.9787 | 1.3638 | 684.0000 | 19.8606 | 156 | 3940 | 3444 | 129 | propia, sólo cobertura; todas las métricas son comunes |
| Random Forest | 0.2270 | 0.5487 | 0.2989 | 0.8666 | 1.1091 | 1 | 0.3030 | 0.6433 | 91 | 0.9794 | 1.3479 | 0.0000 | 0.0000 | 156 | 3940 | 3444 | 129 | propia, sólo cobertura; todas las métricas son comunes |
| XGBoost | 0.2252 | 0.5489 | 0.2984 | 0.9025 | 1.1102 | 1 | 0.2972 | 0.6439 | 91 | 1.0006 | 1.3692 | 0.0000 | 0.0000 | 156 | 3940 | 3444 | 129 | propia, sólo cobertura; todas las métricas son comunes |

Mínimos entre **sólo los tres ML**:

| Modelo | Metrica | Series_minimo_exclusivo | Series_minimo_compartido | Series_minimo_incluyendo_empates | Series_totales |
| --- | --- | --- | --- | --- | --- |
| Regresión Lineal | MAE | 22 | 0 | 22 | 129 |
| Random Forest | MAE | 79 | 0 | 79 | 129 |
| XGBoost | MAE | 28 | 0 | 28 | 129 |
| Regresión Lineal | RMSE | 21 | 0 | 21 | 129 |
| Random Forest | RMSE | 75 | 0 | 75 | 129 |
| XGBoost | RMSE | 33 | 0 | 33 | 129 |

No hay puntajes, pesos de negocio inventados ni ranking por complejidad. Las pequeñas diferencias numéricas no se presentan como estadísticamente significativas.

## J. Por qué el promedio global puede ocultar las semanas positivas

El MAE global es p0·MAE0 + p+·MAE+. El MSE global tiene la misma descomposición con RMSE²; no se promedian directamente los RMSE. Los pesos son 0.876597 y 0.123403. Las contribuciones exactas están en `contribuciones_error.csv`.

Para XGBoost menos RL, el componente de ceros cambia el MAE global en -0.008120; el componente positivo lo cambia en +0.002714. El total es -0.005406: su ventaja de MAE global se explica aritméticamente por la reducción de error absoluto en ceros, a pesar de un MAE mayor en positivas. Esto describe esta muestra, no un mecanismo causal del algoritmo.

RL tiene menor error cuadrático agregado que XGBoost tanto en semanas cero como positivas; por eso mantiene menor RMSE y mayor R² global frente a XGBoost. RF mejora RMSE en positivas respecto de RL, pero su error cuadrático sobre ceros es mayor y el balance global favorece RL. En la misma población, el orden por R² y RMSE no aporta dos evidencias independientes, pues ambos dependen de la misma suma de errores cuadrados.

## K. Limitaciones y reproducibilidad

- Clasificación fija usando TRAIN; puede cambiar el régimen en TEST. Sólo una serie suave, ninguna errática y 60/129 series sin CV² estimable. No se fuerza un contraste equilibrado.
- ADI se estima como N/N+, no como media de distancias entre eventos descartando extremos; CV² usa ddof=1. Cerca del umbral o con pocas positivas, otras convenciones podrían cambiar la etiqueta. No se afinó la convención en función de los errores.
- No se usan periodos futuros para clasificar. La selección previa de la población común depende de cobertura/éxito de los modelos; es una población condicionada.
- ARIMA fijo multi-step y ML secuencial conservan protocolos distintos. La población común no permite atribuir diferencias sólo al algoritmo.
- Semanas extremas potencialmente parciales y ceros interiores bajo continuidad del ERP; cantidad observada no equivale a demanda latente ni mezcla cajas con unidades. Eliminar negativos conserva la definición del experimento base.
- Métricas microponderadas por observación, dependencia temporal y entre series, sin intervalos de confianza ni contrastes estadísticos. Correlaciones y conteos no establecen causalidad.
- La evaluación de negativas no modifica las predicciones. Una decisión operativa necesita definir costos, nivel de servicio, inventario y horizonte; no se inventaron esos criterios.
- Análisis posterior al holdout: una selección resultante requiere validación futura independiente.

Reproducir sin entrenar: `python scripts/analizar_intermitencia.py`; pruebas: `python -m unittest discover -p "test_*.py"`; verificaciones: `python scripts/validar_analisis_intermitencia.py` y `python scripts/validar_evidencia_semanal.py`. El notebook 08 usa este mismo análisis y no llama al runner de entrenamiento. `experimento_intermitencia.json` registra método, versiones, umbrales, cortes, hashes y ausencia de entrenamiento. `docs/integridad_previa_intermitencia.json` protege los archivos del experimento base, incluidos modelos, predicciones, RAW, resultados históricos y app.py. Las figuras PNG son de 300 dpi y muestran series/casos de cada subconjunto.

## Archivos generados y modificados

Se actualizó únicamente el notebook 08 y los atributos de checkout entre los archivos previos permitidos. Se añadieron `src/analisis_intermitencia.py`, `src/informe_intermitencia.py`, `scripts/analizar_intermitencia.py`, `scripts/validar_analisis_intermitencia.py` y `test_analisis_intermitencia.py`. La evidencia anterior queda protegida por `docs/integridad_previa_intermitencia.json`.

En `resultados/semanal/` se generaron:

- `clasificacion_intermitencia.csv`
- `metricas_por_intermitencia.csv`
- `metricas_por_grupo_demanda.csv`
- `metricas_por_ceros.csv`
- `metricas_cero_vs_positivo.csv`
- `auditoria_predicciones_negativas.csv`
- `metricas_por_serie.csv`
- `conteo_minimos_por_serie.csv`
- `conteo_minimos_ml_por_serie.csv`
- `conteo_minimos_por_tipo_serie.csv`
- `contribuciones_error.csv`
- `evaluacion_despliegue.csv`
- `distribucion_intermitencia.csv`
- `asociacion_ceros_error.csv`

Además, `experimento_intermitencia.json`, cinco PNG de 300 dpi en `figuras_intermitencia/` y este informe automático. Se conserva el experimento previo sin sobrescribirlo. No se hizo merge a main.

## Borrador para Resultados

Se analizaron 3444 observaciones comunes de 129 series sucursal-producto. Según TRAIN, se identificaron 1 serie suave, 63 intermitentes y 5 lumpy; no se observaron erráticas. Otras 23 series tuvieron una sola semana positiva y 37 ninguna, por lo que no se estimó CV². El test incluyó 3019 semanas cero (87.66%) y 425 positivas. En las series intermitentes, XGBoost registró MAE 0.3727 y RL RMSE 0.7201; en lumpy, RF obtuvo MAE 0.4589 y RMSE 0.7318. En las semanas positivas, los MAE de RL, RF, XGBoost y ARIMA fueron 0.9787, 0.9794, 1.0007 y 1.0306; los RMSE fueron 1.3638, 1.3479, 1.3692 y 1.3931. Se registraron 684 predicciones negativas de RL y 1048 de ARIMA, frente a ninguna de RF/XGBoost. Las métricas globales comunes permanecieron inalteradas.

## Borrador para Discusión

En el conjunto evaluado, el menor MAE global de XGBoost coexistió con mayor MAE en semanas positivas que RL y RF. La descomposición del error sugiere que la frecuencia de ceros y su menor error absoluto en esas semanas explican aritméticamente la ventaja global. RL obtuvo menor RMSE y mayor R² global, mientras RF mostró menor RMSE cuando existió demanda positiva y mejor MAE/RMSE en las cinco series lumpy. Estos resultados sugieren un comportamiento dependiente del patrón y del criterio de error, sin sustentar superioridad universal. La taxonomía ADI/CV² se apoya en Syntetos et al. (2005); su transferencia a este contexto sirve como descripción y no como regla automática de selección. Cualquier explicación general sobre mecanismos de aprendizaje de los algoritmos ante ceros requiere validación externa [CITA NECESARIA]. La categoría de menor intermitencia cuenta con una sola serie, lo cual impide generalizar el contraste con demanda continua. ARIMA obtuvo numerosos mínimos por serie, pero ello debe interpretarse junto con la abundancia de series de cero, su menor cobertura y su protocolo de origen fijo. ML cubrió 27 series adicionales, pero esa cobertura no prueba mayor exactitud. El análisis es descriptivo, condicionado a la población común y a un holdout ya observado.

## Borrador para Conclusiones

La comparación de RL, RF, XGBoost y ARIMA sobre las mismas observaciones mostró fortalezas complementarias. XGBoost conservó el menor MAE global y RL el menor RMSE/mayor R²; RF presentó menor RMSE en semanas positivas y menor MAE/RMSE en el reducido grupo lumpy. ARIMA mantuvo utilidad como referencia científica, con diferencias de protocolo y cobertura que limitan la atribución de resultados al algoritmo. La elevada proporción de ceros y la escasez de patrones no intermitentes restringen la generalización. La conclusión científica es que ningún modelo domina simultáneamente todos los criterios estudiados. La decisión de implementación queda separada y requiere priorizar objetivos operativos y confirmar su desempeño prospectivamente.

## Evidence for deployment decision

**Recomendación técnica: no sustituir todavía el motor de Streamlit.** Evaluar posteriormente el siguiente compromiso, con los tres ML sobre las mismas observaciones y con la misma cobertura propia:

- **RL:** menor RMSE global (0.5396), mayor R² (0.3219), menor MAE en positivas (0.9787) y menor RMSE en intermitentes; genera 684 negativas (19.8606%).
- **RF:** cero negativas observadas, menor RMSE en positivas (1.3479), mejor MAE/RMSE en lumpy y MAE positivo cercano al de RL. Es un candidato operativo razonable si se priorizan errores grandes en demanda efectiva y salidas no negativas, pero no domina el MAE global ni el RMSE global.
- **XGBoost:** menor MAE global (0.2252), en ceros (0.1160) y en el grupo de mayor intermitencia (0.2972), sin negativas observadas; MAE/RMSE en positivas mayores que RL y RF.

No se declara un ganador operativo automático. La decisión depende de una prioridad de negocio aún no fijada; no se creó un puntaje compuesto y `app.py` permanece intacto.
