# Significancia estadística del MAE por serie

## Datos y unidad de análisis

Se utiliza exclusivamente `resultados/semanal/metricas_por_serie.csv` para la inferencia: 129 series sucursal-producto, cuatro MAE pareados por serie. Los CSV de predicciones comunes y comparación global sólo validan consistencia. No se reentrenan modelos, no se cambian datos, predicciones ni métricas globales. No se usan las 3444 semanas como muestras independientes: dentro de cada serie existe dependencia temporal.

Cada serie recibe igual peso. Las métricas globales agregan errores de cada observación semanal, mientras las pruebas comparan rangos/diferencias entre MAE de series. La magnitud de los errores también importa al MAE agregado, pero Friedman usa rangos dentro de cada serie. Por ello XGBoost puede tener menor MAE agregado y ARIMA menor rango medio, sin contradicción: se describe heterogeneidad del desempeño, no un ganador universal.

## Friedman y tamaño de efecto

Se contrasta la ausencia de diferencias globales entre los cuatro modelos en los mismos bloques pareados. Se usa `scipy.stats.friedmanchisquare`, corrección de empates de SciPy y p asintótico chi-cuadrado con 3 grados de libertad. Kendall W = chi2 / (129 * 3) es el tamaño de efecto global; no se interpreta como porcentaje de error explicado ni como superioridad práctica.

| Prueba | Variable | Series | Modelos | Chi2 | gl | p_value | Kendall_W | alpha | Significativo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Friedman | MAE por serie | 129 | 4 | 57.809302 | 3 | 1.7263748e-12 | 0.14937804 | 0.05 | True |

## Rangos medios

Menor MAE recibe rango 1; los empates exactos reciben rango promedio. Se conserva toda la precisión de los CSV, sin redondear antes de las pruebas.

| Modelo | Rango_medio | Series |
| --- | --- | --- |
| ARIMA (Optimizado ADF/AIC) | 1.9147287 | 129 |
| Random Forest | 2.2635659 | 129 |
| XGBoost | 2.8372093 | 129 |
| Regresión Lineal | 2.9844961 | 129 |

## Wilcoxon bilateral y Holm

Sólo se realizan post-hoc si Friedman es significativo a alpha=0.05. Se comparan los seis pares con Wilcoxon bilateral de rangos con signo. Se pasan las diferencias MAE_A - MAE_B en precisión completa; diferencias exactamente cero se excluyen de los rangos (`zero_method='wilcox'`), sin eliminar la serie del conjunto original. La columna Series cuenta pares válidos y Pares_no_cero cuenta los efectivos. W es el menor de las sumas de rangos positivos y negativos. Se fija aproximación normal `method='asymptotic'`, sin corrección de continuidad; no es un p exacto. Si todos los pares fueran cero se informa W=0, p=1.

Holm ajusta conjuntamente los seis p-values mediante `multipletests(method='holm')` para controlar el error familiar (FWER). La decisión utiliza estrictamente p_holm < 0.05, nunca el p sin ajustar. Holm no elimina las limitaciones de los p-values de entrada.

| Modelo_A | Modelo_B | Series | Pares_no_cero | Diferencias_cero | W | p_raw | p_holm | Significativo_Holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARIMA (Optimizado ADF/AIC) | Regresión Lineal | 129 | 129 | 0 | 2433 | 3.5344226e-05 | 0.00021206536 | True |
| ARIMA (Optimizado ADF/AIC) | Random Forest | 129 | 129 | 0 | 3202 | 0.01989453 | 0.030051161 | True |
| ARIMA (Optimizado ADF/AIC) | XGBoost | 129 | 129 | 0 | 2795 | 0.0010123333 | 0.0040493333 | True |
| Regresión Lineal | Random Forest | 129 | 129 | 0 | 2529 | 9.2174912e-05 | 0.00046087456 | True |
| Regresión Lineal | XGBoost | 129 | 129 | 0 | 2958 | 0.0037092386 | 0.011127716 | True |
| Random Forest | XGBoost | 129 | 129 | 0 | 3158 | 0.01502558 | 0.030051161 | True |

Comparaciones significativas después de Holm: **6 de 6**. Un p ajustado menor que 0.05 indica evidencia de diferencias bajo los supuestos de la prueba. Significancia estadística no implica automáticamente superioridad práctica. No se selecciona ganador.

## Supuestos y límites

Los bloques se tratan como independientes; series del mismo producto o sucursal pueden compartir shocks, por lo que esa independencia no está garantizada. Agregar por serie evita contar semanas como réplicas independientes, pero no resuelve dependencia entre series. Wilcoxon requiere una distribución simétrica de diferencias para su interpretación como contraste de localización; ese supuesto no queda demostrado por obtener significancia. Los p-values de Friedman y Wilcoxon son aproximaciones asintóticas. Con cuatro modelos, debe reconocerse la limitación de la aproximación de Friedman, cuya documentación recomienda más de seis medidas repetidas para su fiabilidad general, además de más de diez bloques.

ARIMA conserva pronóstico multi-step desde origen fijo; ML usa historia disponible semanalmente. Población idéntica no implica protocolo idéntico: las diferencias no pueden atribuirse exclusivamente al algoritmo. El TEST ya se examinó en análisis anteriores; este análisis es posterior y no una validación externa nueva. No se evalúan costes, disponibilidad, stock-outs ni significancia práctica. No se modifican Discusión, Conclusiones, Abstract ni dashboard.

## Integridad y reproducción

Se verifican claves idénticas, 129 series, cuatro modelos, MAE finito, 3444 semanas comunes por modelo, domingos W-SUN y comienzo de cada semana desde 2026-01-01. Se recalculan MAE por serie y métricas globales exclusivamente para validarlas contra sus CSV. Los hashes SHA256 de los 201 archivos previos se comprueban antes y después. Sólo para `.gitattributes` se normalizan finales de línea LF/CRLF impuestos por Git; todos los archivos experimentales se verifican byte a byte. Las métricas globales permanecen exactamente iguales en sus archivos originales.

Ejecutar desde la raíz: `python scripts/analizar_significancia_modelos.py`. Pruebas: `python -m unittest test_significancia_modelos -v`. El script sólo regenera los tres CSV nuevos, este documento y el manifiesto nuevo. La instantánea de integridad inicial no se regenera.

## Referencias técnicas

- [SciPy: Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).
- [SciPy: Wilcoxon, ceros y precisión numérica](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
- [statsmodels: multipletests y Holm](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
