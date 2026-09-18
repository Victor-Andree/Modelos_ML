"""Auditoría independiente de las salidas de notebooks 06, 07 y 08."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experimento_semanal import SALIDA, VERSION, CORTE, sha, base_manifiesto
from src.preprocessing.clean_data import depurar_demanda
from src.preprocessing.demanda_semanal import (
    construir_semanal, agregar_features, excluir_semana_corte, particiones_por_semana, CLAVES, NUMERICAS)
from src.evaluacion_semanal import ARCHIVOS
from src.Modelos.ArimaPredictor import ArimaPredictor
KEY = CLAVES + ['semana']

def leer(nombre, fechas=None):
    return pd.read_csv(SALIDA / nombre, parse_dates=fechas, float_precision='round_trip')

def claves(datos):
    return set(datos[KEY].itertuples(index=False, name=None))

def comprobar_metricas(datos, fila):
    error = datos.y_real.to_numpy() - datos.y_pred.to_numpy()
    denominador = np.square(datos.y_real.to_numpy() - datos.y_real.mean()).sum()
    sse = np.square(error).sum()
    r2 = 1 - sse / denominador if denominador else (1.0 if sse == 0 else 0.0)
    esperadas = [np.abs(error).mean(), np.sqrt(np.square(error).mean()), r2]
    np.testing.assert_allclose(esperadas, [fila.MAE, fila.RMSE, fila.R2], rtol=1e-12, atol=1e-12)
    assert len(datos) == fila.Observaciones_evaluadas
    assert len(datos[CLAVES].drop_duplicates()) == fila.Series_evaluadas

def validar():
    raw = pd.read_csv(ROOT / 'datasets/dataset_maestro_dashboard.csv', parse_dates=['fecdoc'])
    original = raw.copy(deep=True)
    limpio, audit = depurar_demanda(raw)
    pd.testing.assert_frame_equal(raw, original)
    assert audit == json.loads((SALIDA / 'auditoria_limpieza.json').read_text(encoding='utf-8'))
    csv_audit = leer('auditoria_limpieza.csv').iloc[0]
    for k,v in audit.items():
        assert (json.loads(csv_audit[k]) if isinstance(v,list) else csv_audit[k]) == v
    assert audit['registros_iniciales'] == audit['registros_finales'] + audit['registros_negativos_eliminados'] + audit['registros_nulos_eliminados']
    assert limpio.cantidad.ge(0).all() and limpio.cantidad.notna().all()
    pd.testing.assert_frame_equal(limpio, leer('dataset_analitico_depurado.csv',['fecdoc']), check_dtype=False)
    semanal = construir_semanal(limpio)
    pd.testing.assert_frame_equal(semanal, ArimaPredictor(raw).preparar_semanal())
    assert semanal.cantidad.ge(0).all() and semanal.cantidad.notna().all()
    assert not semanal.duplicated(KEY).any()
    # Independent weekly sums from positive/zero transactions, including interior gaps.
    directo = raw.loc[raw.cantidad.notna() & raw.cantidad.ge(0), KEY[:2] + ['fecdoc','cantidad']].copy()
    directo['semana'] = directo.fecdoc.dt.to_period('W-SUN').dt.end_time.dt.normalize()
    sumas = directo.groupby(KEY, observed=True).cantidad.sum()
    observado = semanal.set_index(KEY).cantidad
    np.testing.assert_allclose(observado.reindex(sumas.index), sumas, rtol=0, atol=1e-12)
    assert observado.loc[~observado.index.isin(sumas.index)].eq(0).all()
    assert np.isclose(semanal.cantidad.sum(), limpio.cantidad.sum())
    for _,grupo in semanal.groupby(CLAVES, observed=True):
        assert grupo.semana.tolist() == pd.date_range(grupo.semana.min(),grupo.semana.max(),freq='W-SUN').tolist()
    for archivo in ['dataset_semanal.csv','dataset_semanal_depurado.csv']:
        pd.testing.assert_frame_equal(semanal, leer(archivo,['semana']), check_dtype=False)
    evaluable, cruzadas = excluir_semana_corte(semanal)
    pd.testing.assert_frame_equal(evaluable.reset_index(drop=True), leer('dataset_semanal_evaluable.csv',['semana']),check_dtype=False)
    pd.testing.assert_frame_equal(cruzadas.reset_index(drop=True), leer('semanas_excluidas_corte.csv',['semana']),check_dtype=False)
    assert (evaluable.loc[evaluable.semana.ge(CORTE),'semana'] - pd.Timedelta(days=6)).ge(pd.Timestamp(CORTE)).all()
    participantes = directo.merge(evaluable.loc[evaluable.semana.ge(CORTE),KEY],on=KEY,validate='many_to_one')
    assert participantes.fecdoc.ge(CORTE).all()
    features = agregar_features(semanal)
    for _, grupo in features.groupby(CLAVES, observed=True):
        for lag in [1,2,4,8]:
            np.testing.assert_allclose(grupo[f'lag_{lag}'],grupo.cantidad.shift(lag),equal_nan=True)
        for ventana in [4,8]:
            np.testing.assert_allclose(grupo[f'rolling_mean_{ventana}'],grupo.cantidad.shift(1).rolling(ventana).mean(),equal_nan=True)
    elegibles, _ = excluir_semana_corte(features.dropna(subset=NUMERICAS))
    elegibles = elegibles.sort_values(['semana'] + CLAVES).reset_index(drop=True)
    train = elegibles.loc[elegibles.semana.lt(CORTE)].reset_index(drop=True)
    test = elegibles.loc[elegibles.semana.ge(CORTE)].reset_index(drop=True)
    pd.testing.assert_frame_equal(train,leer('dataset_train_ml.csv',['semana']),check_dtype=False)
    pd.testing.assert_frame_equal(test,leer('dataset_test_ml.csv',['semana']),check_dtype=False)
    assert (train.semana + pd.Timedelta(days=1)).le(pd.Timestamp(CORTE)).all()
    for a,b in particiones_por_semana(train.semana):
        assert train.semana.iloc[a].max() < train.semana.iloc[b].min()
    predicciones = {}
    propia = leer('metricas_cobertura_propia.csv').set_index('Modelo')
    assert set(propia.index) == set(ARCHIVOS) and propia.index.is_unique
    for nombre,archivo in ARCHIVOS.items():
        d = leer(archivo,['semana'])
        assert not d.duplicated(KEY).any()
        assert d[KEY + ['y_real','y_pred']].notna().all().all()
        assert np.isfinite(d[['y_real','y_pred']]).all().all() and d.y_real.ge(0).all()
        assert (d.semana-pd.Timedelta(days=6)).ge(pd.Timestamp(CORTE)).all()
        validacion = d.merge(evaluable[KEY + ['cantidad']],on=KEY,validate='one_to_one')
        assert len(validacion) == len(d)
        np.testing.assert_array_equal(validacion.y_real,validacion.cantidad)
        if not nombre.startswith('ARIMA'):
            assert claves(d) == claves(test)
        else:
            origen = pd.to_datetime(d.origen)
            assert origen.lt(CORTE).all()
            np.testing.assert_array_equal((d.semana-origen).dt.days // 7,d.horizonte_semanas)
        np.testing.assert_allclose(d.y_real-d.y_pred,d.error,atol=1e-12)
        np.testing.assert_allclose(d.error.abs(),d.error_absoluto,atol=1e-12)
        comprobar_metricas(d,propia.loc[nombre])
        tabla_origen = leer('metricas_arima.csv' if nombre.startswith('ARIMA') else 'metricas_ml_semanal.csv').set_index('Modelo')
        comprobar_metricas(d, tabla_origen.loc[nombre])
        predicciones[nombre] = d
    interseccion = set.intersection(*(claves(d) for d in predicciones.values()))
    comun = leer('predicciones_comunes.csv',['semana'])
    tabla = leer('comparacion_modelos_comun.csv').set_index('Modelo')
    assert set(comun.Modelo) == set(tabla.index) == set(ARCHIVOS)
    assert tabla.index.is_unique and tabla.Poblacion.eq('comun').all()
    assert not comun.duplicated(['Modelo'] + KEY).any()
    assert comun.groupby('Modelo').size().nunique() == 1
    assert tabla.Observaciones_evaluadas.nunique() == tabla.Series_evaluadas.nunique() == 1
    reales = []
    for nombre in ARCHIVOS:
        d = comun.loc[comun.Modelo.eq(nombre)].sort_values(KEY).reset_index(drop=True)
        assert claves(d) == interseccion and len(d) == len(interseccion)
        original = predicciones[nombre].set_index(KEY).loc[d.set_index(KEY).index]
        np.testing.assert_array_equal(d.y_pred,original.y_pred)
        np.testing.assert_array_equal(d.y_real,original.y_real)
        reales.append(d.y_real.to_numpy())
        comprobar_metricas(d,tabla.loc[nombre])
    for valores in reales[1:]: np.testing.assert_array_equal(valores,reales[0])
    cobertura = leer('resumen_cobertura.csv').set_index('Modelo')
    assert cobertura.Observaciones_comunes.eq(len(interseccion)).all()
    assert cobertura.Series_comunes.eq(tabla.Series_evaluadas.iloc[0]).all()
    for nombre,d in predicciones.items():
        assert cobertura.loc[nombre,'Observaciones_evaluadas'] == len(d)
        assert cobertura.loc[nombre,'Observaciones_fuera_comun'] == len(d)-len(interseccion)
    pd.testing.assert_frame_equal(leer('comparacion_modelos.csv'),leer('comparacion_modelos_comun.csv'))
    arima_registro = leer('registro_series_arima.csv')
    conteo = json.loads((SALIDA / 'conteo_arima.json').read_text(encoding='utf-8'))
    assert len(arima_registro) == conteo['encontradas']
    assert not arima_registro.duplicated(CLAVES).any()
    for estado,k in [('modelada','modeladas'),('descartada','descartadas'),('fallida','fallidas')]:
        assert arima_registro.estado.eq(estado).sum() == conteo[k]
    assert arima_registro.loc[arima_registro.estado.ne('modelada'),'motivo'].notna().all()
    assert conteo['encontradas'] == sum(conteo[k] for k in ['descartadas','fallidas','modeladas'])
    actual = base_manifiesto()
    for archivo in ['experimento_ml.json','experimento_arima.json','experimento_comun.json']:
        m = json.loads((SALIDA / archivo).read_text(encoding='utf-8'))
        for k in ['version_target','sha256_dataset','sha256_analitico','sha256_constructor','sha256_limpieza','codigo_sha256']:
            assert m[k] == actual[k], f'{archivo}: {k}'
    m = json.loads((SALIDA / 'experimento_comun.json').read_text(encoding='utf-8'))
    assert m['observaciones_comunes'] == len(interseccion)
    for archivo,huella in {**m['fuentes_sha256'], **m['resultados_sha256']}.items():
        assert sha(SALIDA / archivo) == huella
    for nombre in ['06_Entrenamiento_ARIMA.ipynb','07_Entrenamiento_MachineLearning.ipynb','08_EDA_Intermitencia_Series.ipynb']:
        nb = nbformat.read(ROOT / 'notebooks' / nombre,as_version=4)
        nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type == 'code':
                assert c.execution_count is not None
                assert not any(o.output_type == 'error' for o in c.outputs)
    print(f'OK: depuración, auditoría, agregación independiente, corte, features causales, folds, cobertura y métricas. {len(interseccion)} observaciones comunes por modelo, {tabla.Series_evaluadas.iloc[0]} series.')
    return tabla.reset_index()

if __name__ == '__main__': validar()
