"""Operaciones reproducibles utilizadas por notebooks 06/07/08 y validadores."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from src.preprocessing.clean_data import depurar_demanda
from src.preprocessing.demanda_semanal import (
    construir_semanal, agregar_features, excluir_semana_corte, resumir_intermitencia, CLAVES, NUMERICAS)
from src.evaluacion_semanal import ARCHIVOS, comparar

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / 'resultados/semanal'
VERSION = 'demanda_observada_sin_negativos_corte_estricto_v2'
CORTE = '2026-01-01'

def sha(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()

def guardar_json(ruta, datos):
    Path(ruta).write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def base_manifiesto():
    return {'version_target': VERSION, 'fecha_corte': CORTE, 'frecuencia': 'W-SUN',
        'target': 'Suma semanal de cantidad ERP tras excluir transacciones NaN y negativas; ceros conservados',
        'sha256_dataset': sha(ROOT / 'datasets/dataset_maestro_dashboard.csv'),
        'sha256_analitico': sha(SALIDA / 'dataset_analitico_depurado.csv'),
        'sha256_constructor': sha(ROOT / 'src/preprocessing/demanda_semanal.py'),
        'sha256_limpieza': sha(ROOT / 'src/preprocessing/clean_data.py'),
        'codigo_sha256': {p: sha(ROOT / p) for p in ['src/evaluacion_semanal.py',
            'src/experimento_semanal.py', 'src/Modelos/ArimaPredictor.py', 'src/Modelos/MachineLearningPipeline.py']},
        'python': platform.python_version(),
        'versiones': {p: importlib.metadata.version(p) for p in
            ['pandas','numpy','statsmodels','scikit-learn','xgboost','joblib']},
        'generado_utc': datetime.now(timezone.utc).isoformat()}

def preparar_datos():
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = ROOT / 'datasets/dataset_maestro_dashboard.csv'
    antes = sha(ruta)
    raw = pd.read_csv(ruta, parse_dates=['fecdoc'])
    limpio, auditoria = depurar_demanda(raw)
    limpio.to_csv(SALIDA / 'dataset_analitico_depurado.csv', index=False)
    guardar_json(SALIDA / 'auditoria_limpieza.json', auditoria)
    pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v,list) else v
                   for k,v in auditoria.items()}]).to_csv(SALIDA / 'auditoria_limpieza.csv', index=False)
    eliminadas = raw.loc[raw.cantidad.isna() | raw.cantidad.lt(0)].copy()
    eliminadas['motivo_exclusion'] = np.where(eliminadas.cantidad.isna(), 'cantidad_nula', 'cantidad_negativa')
    eliminadas.to_csv(SALIDA / 'transacciones_excluidas.csv', index=False)
    semanal = construir_semanal(limpio)
    evaluable, excluidas = excluir_semana_corte(semanal, CORTE)
    for nombre, datos in [('dataset_semanal.csv', semanal), ('dataset_semanal_depurado.csv', semanal),
                          ('dataset_semanal_evaluable.csv', evaluable), ('semanas_excluidas_corte.csv', excluidas)]:
        datos.to_csv(SALIDA / nombre, index=False)
    con_features = agregar_features(semanal)
    features, _ = excluir_semana_corte(con_features, CORTE)
    listo = features.dropna(subset=NUMERICAS).sort_values(['semana'] + CLAVES).reset_index(drop=True)
    train = listo.loc[listo.semana.lt(CORTE)].copy()
    test = listo.loc[listo.semana.ge(CORTE)].copy()
    assert len(train) and len(test)
    assert (test.semana - pd.Timedelta(days=6)).ge(pd.Timestamp(CORTE)).all()
    train.to_csv(SALIDA / 'dataset_train_ml.csv', index=False)
    test.to_csv(SALIDA / 'dataset_test_ml.csv', index=False)
    # Audit series which disappear when all their transactions were removed.
    originales = raw.groupby(CLAVES, observed=True).size().rename('registros_raw').reset_index()
    finales = limpio.groupby(CLAVES, observed=True).size().rename('registros_analiticos').reset_index()
    serie_audit = originales.merge(finales, on=CLAVES, how='left').fillna({'registros_analiticos':0})
    serie_audit.to_csv(SALIDA / 'auditoria_series_limpieza.csv', index=False)
    resumen = {**auditoria, 'observaciones_semanales': len(semanal),
        'observaciones_excluidas_corte': len(excluidas), 'observaciones_evaluables': len(evaluable),
        'series_semanales': len(semanal[CLAVES].drop_duplicates()),
        'train_antes_lags': int(evaluable.semana.lt(CORTE).sum()),
        'test_antes_lags': int(evaluable.semana.ge(CORTE).sum()),
        'eliminadas_por_lags': len(features) - len(listo),
        'eliminadas_por_lags_calendario_completo': int(con_features[NUMERICAS].isna().any(axis=1).sum()),
        'train_modelo': len(train), 'test_modelo': len(test),
        'series_train': len(train[CLAVES].drop_duplicates()), 'series_test': len(test[CLAVES].drop_duplicates())}
    guardar_json(SALIDA / 'resumen_dataset.json', resumen)
    inter = resumir_intermitencia(semanal)
    inter.to_csv(SALIDA / 'intermitencia_semanal_completa.csv', index=False)
    resumir_intermitencia(evaluable).to_csv(SALIDA / 'intermitencia_semanal.csv', index=False)
    semanal.cantidad.describe().to_csv(SALIDA / 'descriptivos_semanales.csv')
    assert antes == sha(ruta), 'Se modificó el original'
    return raw, limpio, semanal, train, test, resumen

def ejecutar_arima():
    from src.Modelos.ArimaPredictor import ArimaPredictor
    raw, limpio, semanal, train, test, resumen = preparar_datos()
    modelo = ArimaPredictor(limpio)
    metricas = modelo.entrenar_y_evaluar()
    pd.testing.assert_frame_equal(modelo.semanal, semanal)
    modelo.predicciones.to_csv(SALIDA / ARCHIVOS['ARIMA (Optimizado ADF/AIC)'], index=False)
    modelo.registro_series.to_csv(SALIDA / 'registro_series_arima.csv', index=False)
    pd.DataFrame([metricas]).to_csv(SALIDA / 'metricas_arima.csv', index=False)
    guardar_json(SALIDA / 'conteo_arima.json', modelo.conteo_series)
    manifiesto = {**base_manifiesto(), 'protocolo': 'multi-step desde origen fijo',
        'observaciones_evaluadas': len(modelo.predicciones), 'resumen_dataset': resumen,
        'conteo_series': modelo.conteo_series, 'minimos': {'total':20,'train':10,'test':2},
        'seleccion': 'ADF para d; AIC p,q=0..3; sin modificación de algoritmos',
        'sha256_predicciones': sha(SALIDA / ARCHIVOS['ARIMA (Optimizado ADF/AIC)'])}
    guardar_json(SALIDA / 'experimento_arima.json', manifiesto)
    return modelo, metricas

def ejecutar_ml():
    from src.Modelos.MachineLearningPipeline import MachineLearningPipeline
    raw, limpio, semanal, train, test, resumen = preparar_datos()
    pipeline = MachineLearningPipeline(train[CLAVES + NUMERICAS], train.cantidad,
        test[CLAVES + NUMERICAS], test.cantidad, train.semana, test[CLAVES + ['semana']])
    folds = []
    for i,(a,b) in enumerate(pipeline.cv,1):
        assert train.semana.iloc[a].max() < train.semana.iloc[b].min()
        folds.append({'fold':i,'filas_train':len(a),'filas_validacion':len(b),
            'train_inicio': str(train.semana.iloc[a].min()), 'train_fin': str(train.semana.iloc[a].max()),
            'validacion_inicio': str(train.semana.iloc[b].min()), 'validacion_fin': str(train.semana.iloc[b].max())})
    pipeline.ejecutar_regresion_lineal()
    pipeline.ejecutar_random_forest()
    pipeline.ejecutar_xgboost()
    tabla = pipeline.obtener_tabla_resultados()
    tabla.to_csv(SALIDA / 'metricas_ml_semanal.csv', index=False)
    for nombre, datos in pipeline.predicciones.items():
        datos.to_csv(SALIDA / ARCHIVOS[nombre], index=False)
    for nombre, datos in pipeline.cv_results_.items():
        datos.to_csv(SALIDA / f'cv_{nombre.replace(" ", "_")}.csv', index=False)
    manifiesto = {**base_manifiesto(), 'dataset_semanal': resumen,
        'protocolo_ml':'una semana secuencial; sin reentrenar durante test',
        'semilla':42, 'n_splits':3, 'folds':folds, 'best_params':pipeline.best_params_,
        'best_score_neg_mse':pipeline.best_score_,
        'sha256_predicciones': {n:sha(SALIDA / ARCHIVOS[n]) for n in pipeline.predicciones}}
    guardar_json(SALIDA / 'experimento_ml.json', manifiesto)
    guardar_json(SALIDA / 'best_params.json', pipeline.best_params_)
    # Export every fitted ML model; there is no hard-coded winner assertion.
    nombres = {'Regresión Lineal':'regresion_lineal', 'Random Forest':'random_forest', 'XGBoost':'xgboost'}
    for nombre, modelo in pipeline.modelos.items():
        ruta = ROOT / f'notebooks/modelo_{nombres[nombre]}_semanal.pkl'
        joblib.dump(modelo, ruta)
        np.testing.assert_allclose(joblib.load(ruta).predict(test[CLAVES + NUMERICAS]),
                                      pipeline.predicciones[nombre].y_pred, rtol=1e-12, atol=1e-12)
        fila = tabla.set_index('Modelo').loc[nombre]
        metadata = {**base_manifiesto(), 'modelo':nombre, 'criterio_seleccion':
            'Artefacto exportado sin declarar ganador; comparar en comparacion_modelos_comun.csv',
            'entrenado_solo_en_train':True, 'train_inicio':str(train.semana.min()),
            'train_fin':str(train.semana.max()), 'columnas_entrada':CLAVES + NUMERICAS,
            'metricas_holdout': {k:float(fila[k]) for k in ['MAE','RMSE','R2']},
            'poblacion_metricas':'propia', 'observaciones_holdout':len(test),
            'sha256_modelo':sha(ruta), 'predicciones_negativas':'Se conservan sin recorte'}
        guardar_json(ruta.with_suffix('.json'), metadata)
    return pipeline, tabla

def ejecutar_comparacion():
    ml = json.loads((SALIDA / 'experimento_ml.json').read_text(encoding='utf-8'))
    ar = json.loads((SALIDA / 'experimento_arima.json').read_text(encoding='utf-8'))
    actual = base_manifiesto()
    for clave in ['version_target','sha256_dataset','sha256_analitico','sha256_constructor','sha256_limpieza','codigo_sha256']:
        if not ml[clave] == ar[clave] == actual[clave]:
            raise ValueError(f'Ejecuciones incompatibles: {clave}. Reejecutar 06 y 07.')
    predicciones = {}
    for nombre, archivo in ARCHIVOS.items():
        esperado = ar['sha256_predicciones'] if nombre.startswith('ARIMA') else ml['sha256_predicciones'][nombre]
        assert esperado == sha(SALIDA / archivo), f'Predicciones alteradas: {nombre}'
        predicciones[nombre] = pd.read_csv(SALIDA / archivo, parse_dates=['semana'], float_precision='round_trip')
    comun, tabla, propias, cobertura = comparar(predicciones)
    for archivo, datos in [('predicciones_comunes.csv',comun), ('comparacion_modelos_comun.csv',tabla),
        ('metricas_cobertura_propia.csv',propias), ('resumen_cobertura.csv',cobertura)]:
        datos.to_csv(SALIDA / archivo, index=False)
    # Explicit alias for previous consumers: now refers to COMMON, never own coverage.
    tabla.to_csv(SALIDA / 'comparacion_modelos.csv', index=False)
    m = {**actual, 'poblacion':'Intersección exacta de predicciones válidas en sucursal, producto, semana',
         'formato_predicciones_comunes':'largo: una fila por clave y modelo',
         'observaciones_comunes': int(tabla.Observaciones_evaluadas.iloc[0]),
         'series_comunes':int(tabla.Series_evaluadas.iloc[0]),
         'limite':'Misma población; protocolos de información distintos: ARIMA fijo multi-step, ML secuencial semanal',
         'fuentes_sha256': {archivo:sha(SALIDA / archivo) for archivo in ARCHIVOS.values()},
         'resultados_sha256': {a:sha(SALIDA / a) for a in ['predicciones_comunes.csv','comparacion_modelos_comun.csv','metricas_cobertura_propia.csv','resumen_cobertura.csv']}}
    guardar_json(SALIDA / 'experimento_comun.json', m)
    return comun, tabla, propias, cobertura
