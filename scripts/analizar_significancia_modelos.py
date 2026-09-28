"""Inferencia pareada por serie desde evidencia congelada; nunca entrena modelos."""
from pathlib import Path
from itertools import combinations
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
MODELOS = ['ARIMA (Optimizado ADF/AIC)', 'Regresión Lineal', 'Random Forest', 'XGBoost']
CLAVES = ['sucursal', 'producto']
ALPHA = 0.05
ESPERADAS = {
    MODELOS[0]: [0.23633302823110328, 0.5796836321886407, 0.21749041943698],
    MODELOS[1]: [0.23059090596656548, 0.539644857287217, 0.3218532683231031],
    MODELOS[2]: [0.22703005641196106, 0.5486983586714412, 0.2989081641123271],
    MODELOS[3]: [0.22518463106609612, 0.5488974093470322, 0.298399403281957],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verificar_integridad(root=ROOT):
    registro = json.loads((root / 'docs/integridad_previa_significancia.json').read_text(encoding='utf-8'))
    for nombre, esperado in registro['archivos'].items():
        assert sha(root / nombre) == esperado, f'Archivo previo alterado: {nombre}'
    return registro


def cargar_validar(root=ROOT):
    """Los datos de inferencia son únicamente MAE del CSV por serie."""
    ruta = root / 'resultados/semanal'
    leer = lambda nombre: pd.read_csv(ruta / nombre, float_precision='round_trip')
    serie = leer('metricas_por_serie.csv')
    pred = leer('predicciones_comunes.csv')
    globales = leer('comparacion_modelos_comun.csv').set_index('Modelo')
    assert set(serie.Modelo) == set(pred.Modelo) == set(globales.index) == set(MODELOS)
    assert len(globales) == 4 and not globales.index.duplicated().any()
    assert len(serie) == 129 * 4
    assert not serie.duplicated(CLAVES + ['Modelo']).any()
    assert not serie[CLAVES].isna().any().any()
    assert np.isfinite(serie.MAE).all() and serie.MAE.ge(0).all()
    matriz = serie.pivot(index=CLAVES, columns='Modelo', values='MAE').reindex(columns=MODELOS).sort_index()
    assert matriz.shape == (129, 4) and np.isfinite(matriz.to_numpy()).all()
    pred['semana'] = pd.to_datetime(pred.semana)
    assert not pred[CLAVES + ['semana']].isna().any().any()
    assert np.isfinite(pred[['y_real', 'y_pred']]).all().all()
    assert pred.y_real.ge(0).all()
    assert pred.semana.dt.dayofweek.eq(6).all()
    assert (pred.semana - pd.Timedelta(days=6)).ge(pd.Timestamp('2026-01-01')).all(), 'Predicciones fuera de TEST estricto'
    keys = CLAVES + ['semana']
    assert not pred.duplicated(keys + ['Modelo']).any()
    referencia = pred.loc[pred.Modelo.eq(MODELOS[0])].set_index(keys).sort_index()
    assert len(referencia) == 3444
    for modelo in MODELOS:
        d = pred.loc[pred.Modelo.eq(modelo)].set_index(keys).sort_index()
        pd.testing.assert_index_equal(d.index, referencia.index)
        np.testing.assert_array_equal(d.y_real, referencia.y_real)
        error = d.y_real - d.y_pred
        np.testing.assert_allclose(d.error, error, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d.error_absoluto, abs(error), rtol=1e-12, atol=1e-12)
        por_serie = abs(error).groupby(level=CLAVES).mean()
        pd.testing.assert_index_equal(por_serie.index, matriz.index)
        np.testing.assert_allclose(por_serie, matriz[modelo], rtol=1e-12, atol=1e-12)
        cantidades = d.groupby(level=CLAVES).size()
        registrada = serie.loc[serie.Modelo.eq(modelo)].set_index(CLAVES).sort_index()
        np.testing.assert_array_equal(cantidades, registrada.Observaciones)
        calculadas = [abs(error).mean(), np.sqrt(np.mean(error**2)), 1 - np.sum(error**2) / np.sum((d.y_real - d.y_real.mean())**2)]
        np.testing.assert_allclose(globales.loc[modelo, ['MAE', 'RMSE', 'R2']].astype(float), calculadas, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(calculadas, ESPERADAS[modelo], rtol=1e-12, atol=1e-12)
        assert globales.loc[modelo, 'Observaciones_evaluadas'] == 3444
        assert globales.loc[modelo, 'Series_evaluadas'] == 129
    return matriz


def calcular(matriz):
    assert matriz.shape == (129, 4) and list(matriz.columns) == MODELOS
    assert matriz.index.is_unique and np.isfinite(matriz.to_numpy()).all()
    resultado = friedmanchisquare(*[matriz[m].to_numpy() for m in MODELOS])
    w = resultado.statistic / (len(matriz) * (len(MODELOS) - 1))
    assert 0 <= w <= 1
    friedman = pd.DataFrame([dict(Prueba='Friedman', Variable='MAE por serie', Series=129, Modelos=4,
        Chi2=resultado.statistic, gl=3, p_value=resultado.pvalue, Kendall_W=w, alpha=ALPHA,
        Significativo=resultado.pvalue < ALPHA)])
    columnas = ['Modelo_A', 'Modelo_B', 'Series', 'W', 'p_raw', 'p_holm', 'alpha', 'Significativo_Holm',
                'Pares_no_cero', 'Diferencias_cero', 'Zero_method', 'Metodo', 'Alternativa', 'Correccion_continuidad']
    filas = []
    if resultado.pvalue < ALPHA:
        for a, b in combinations(MODELOS, 2):
            diferencias = (matriz[a] - matriz[b]).to_numpy()
            ceros = int(np.count_nonzero(diferencias == 0))
            # Mantener precisión CSV: no redondear ni introducir tolerancias para empates.
            if ceros == len(diferencias):
                estadistico, p = 0.0, 1.0
            else:
                estadistico, p = wilcoxon(diferencias, zero_method='wilcox', correction=False,
                                          alternative='two-sided', method='asymptotic')
            filas.append(dict(Modelo_A=a, Modelo_B=b, Series=len(diferencias), W=estadistico, p_raw=p,
                alpha=ALPHA, Pares_no_cero=len(diferencias)-ceros, Diferencias_cero=ceros,
                Zero_method='wilcox', Metodo='asymptotic', Alternativa='two-sided', Correccion_continuidad=False))
        ajustados = multipletests([f['p_raw'] for f in filas], alpha=ALPHA, method='holm')[1]
        for fila, ajustado in zip(filas, ajustados):
            fila.update(p_holm=ajustado, Significativo_Holm=bool(ajustado < ALPHA))
            assert ajustado >= fila['p_raw']
    posthoc = pd.DataFrame(filas, columns=columnas)
    rangos = matriz.rank(axis=1, method='average', ascending=True).mean().rename('Rango_medio').rename_axis('Modelo').reset_index()
    rangos['Series'] = len(matriz)
    return friedman, posthoc, rangos.sort_values('Rango_medio', ignore_index=True)


def tabla(df):
    def formato(x):
        return f'{x:.8g}' if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| ' + ' | '.join(df.columns) + ' |', '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'] +
                     ['| ' + ' | '.join(map(formato, fila)) + ' |' for fila in df.itertuples(index=False, name=None)])


def ejecutar(root=ROOT):
    registro = verificar_integridad(root)
    matriz = cargar_validar(root)
    friedman, posthoc, rangos = calcular(matriz)
    ruta = root / 'resultados/semanal'
    salidas = {'significancia_friedman.csv': friedman, 'significancia_wilcoxon_holm.csv': posthoc,
               'rangos_mae_por_modelo.csv': rangos}
    for nombre, datos in salidas.items():
        (ruta / nombre).write_bytes(datos.to_csv(index=False, lineterminator='\n').encode('utf-8'))
    f = friedman.iloc[0]
    reporte = f'''# Significancia estadística del MAE por serie

## Datos y unidad de análisis

Se utiliza exclusivamente `resultados/semanal/metricas_por_serie.csv` para la inferencia: 129 series sucursal-producto, cuatro MAE pareados por serie. Los CSV de predicciones comunes y comparación global sólo validan consistencia. No se reentrenan modelos, no se cambian datos, predicciones ni métricas globales. No se usan las 3444 semanas como muestras independientes: dentro de cada serie existe dependencia temporal.

Cada serie recibe igual peso. Las métricas globales agregan errores de cada observación semanal, mientras las pruebas comparan rangos/diferencias entre MAE de series. La magnitud de los errores también importa al MAE agregado, pero Friedman usa rangos dentro de cada serie. Por ello XGBoost puede tener menor MAE agregado y ARIMA menor rango medio, sin contradicción: se describe heterogeneidad del desempeño, no un ganador universal.

## Friedman y tamaño de efecto

Se contrasta la ausencia de diferencias globales entre los cuatro modelos en los mismos bloques pareados. Se usa `scipy.stats.friedmanchisquare`, corrección de empates de SciPy y p asintótico chi-cuadrado con 3 grados de libertad. Kendall W = chi2 / (129 * 3) es el tamaño de efecto global; no se interpreta como porcentaje de error explicado ni como superioridad práctica.

{tabla(friedman)}

## Rangos medios

Menor MAE recibe rango 1; los empates exactos reciben rango promedio. Se conserva toda la precisión de los CSV, sin redondear antes de las pruebas.

{tabla(rangos)}

## Wilcoxon bilateral y Holm

Sólo se realizan post-hoc si Friedman es significativo a alpha=0.05. Se comparan los seis pares con Wilcoxon bilateral de rangos con signo. Se pasan las diferencias MAE_A - MAE_B en precisión completa; diferencias exactamente cero se excluyen de los rangos (`zero_method='wilcox'`), sin eliminar la serie del conjunto original. La columna Series cuenta pares válidos y Pares_no_cero cuenta los efectivos. W es el menor de las sumas de rangos positivos y negativos. Se fija aproximación normal `method='asymptotic'`, sin corrección de continuidad; no es un p exacto. Si todos los pares fueran cero se informa W=0, p=1.

Holm ajusta conjuntamente los seis p-values mediante `multipletests(method='holm')` para controlar el error familiar (FWER). La decisión utiliza estrictamente p_holm < 0.05, nunca el p sin ajustar. Holm no elimina las limitaciones de los p-values de entrada.

{tabla(posthoc[['Modelo_A','Modelo_B','Series','Pares_no_cero','Diferencias_cero','W','p_raw','p_holm','Significativo_Holm']])}

Comparaciones significativas después de Holm: **{int(posthoc.Significativo_Holm.sum())} de {len(posthoc)}**. Un p ajustado menor que 0.05 indica evidencia de diferencias bajo los supuestos de la prueba. Significancia estadística no implica automáticamente superioridad práctica. No se selecciona ganador.

## Supuestos y límites

Los bloques se tratan como independientes; series del mismo producto o sucursal pueden compartir shocks, por lo que esa independencia no está garantizada. Agregar por serie evita contar semanas como réplicas independientes, pero no resuelve dependencia entre series. Wilcoxon requiere una distribución simétrica de diferencias para su interpretación como contraste de localización; ese supuesto no queda demostrado por obtener significancia. Los p-values de Friedman y Wilcoxon son aproximaciones asintóticas. Con cuatro modelos, debe reconocerse la limitación de la aproximación de Friedman, cuya documentación recomienda más de seis medidas repetidas para su fiabilidad general, además de más de diez bloques.

ARIMA conserva pronóstico multi-step desde origen fijo; ML usa historia disponible semanalmente. Población idéntica no implica protocolo idéntico: las diferencias no pueden atribuirse exclusivamente al algoritmo. El TEST ya se examinó en análisis anteriores; este análisis es posterior y no una validación externa nueva. No se evalúan costes, disponibilidad, stock-outs ni significancia práctica. No se modifican Discusión, Conclusiones, Abstract ni dashboard.

## Integridad y reproducción

Se verifican claves idénticas, 129 series, cuatro modelos, MAE finito, 3444 semanas comunes por modelo, domingos W-SUN y comienzo de cada semana desde 2026-01-01. Se recalculan MAE por serie y métricas globales exclusivamente para validarlas contra sus CSV. Los hashes SHA256 de los {len(registro['archivos'])} archivos previos se comprueban antes y después. Las métricas globales permanecen exactamente iguales en sus archivos originales.

Ejecutar desde la raíz: `python scripts/analizar_significancia_modelos.py`. Pruebas: `python -m unittest test_significancia_modelos -v`. El script sólo regenera los tres CSV nuevos, este documento y el manifiesto nuevo. La instantánea de integridad inicial no se regenera.

## Referencias técnicas

- [SciPy: Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).
- [SciPy: Wilcoxon, ceros y precisión numérica](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
- [statsmodels: multipletests y Holm](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
'''
    informe = root / 'docs/SIGNIFICANCIA_ESTADISTICA.md'
    informe.write_bytes(reporte.encode('utf-8'))
    manifest = {'version': 'significancia_mae_serie_v1', 'commit_base': registro['commit_base'],
        'versiones': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'statsmodels']},
        'alpha': ALPHA, 'wilcoxon': {'method': 'asymptotic', 'zero_method': 'wilcox', 'correction': False, 'alternative': 'two-sided'},
        'entradas_sha256': {n: sha(ruta/n) for n in ['metricas_por_serie.csv','predicciones_comunes.csv','comparacion_modelos_comun.csv']},
        'salidas_sha256': {n: sha(ruta/n) for n in salidas}, 'informe_sha256': sha(informe),
        'codigo_sha256': sha(root/'scripts/analizar_significancia_modelos.py')}
    (ruta/'manifiesto_significancia.json').write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    verificar_integridad(root)
    print(tabla(friedman)); print(tabla(rangos)); print(tabla(posthoc))
    print(f'Integridad OK: {len(registro["archivos"])} archivos previos intactos.')
    return friedman, posthoc, rangos


if __name__ == '__main__':
    ejecutar()
