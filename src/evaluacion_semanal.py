"""Métricas de cobertura propia y población común, calculadas desde predicciones."""
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CLAVES = ['sucursal', 'producto', 'semana']
ARCHIVOS = {
    'Regresión Lineal': 'predicciones_Regresión_Lineal_semanal.csv',
    'Random Forest': 'predicciones_Random_Forest_semanal.csv',
    'XGBoost': 'predicciones_XGBoost_semanal.csv',
    'ARIMA (Optimizado ADF/AIC)': 'predicciones_arima_concatenadas.csv',
}

def metricas_predicciones(datos):
    if len(datos) < 2 or not np.isfinite(datos[['y_real', 'y_pred']]).all().all():
        raise ValueError('Se necesitan al menos dos predicciones finitas')
    return {'MAE': float(mean_absolute_error(datos.y_real, datos.y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(datos.y_real, datos.y_pred))),
            'R2': float(r2_score(datos.y_real, datos.y_pred))}

def comparar(predicciones):
    if set(predicciones) != set(ARCHIVOS):
        raise ValueError('Se requieren los cuatro modelos')
    validas, cobertura = {}, []
    for nombre, original in predicciones.items():
        datos = original.copy()
        datos['semana'] = pd.to_datetime(datos.semana, errors='raise')
        if datos[CLAVES].isna().any().any() or datos.duplicated(CLAVES).any():
            raise ValueError(f'Claves nulas o duplicadas: {nombre}')
        finitas = np.isfinite(datos[['y_real', 'y_pred']]).all(axis=1)
        datos = datos.loc[finitas].copy()
        assert datos.y_real.ge(0).all()
        assert (datos.semana - pd.Timedelta(days=6)).ge(pd.Timestamp('2026-01-01')).all()
        validas[nombre] = datos
        cobertura.append({'Modelo': nombre, 'Predicciones_originales': len(original),
            'Predicciones_no_finitas': int((~finitas).sum())})
    claves = reduce(lambda a,b: a.merge(b, on=CLAVES, how='inner', validate='one_to_one'),
                    [d[CLAVES] for d in validas.values()]).sort_values(CLAVES).reset_index(drop=True)
    if len(claves) < 2:
        raise ValueError('Intersección insuficiente para métricas')
    comunes, propias, tablas = [], [], []
    reales = None
    for nombre, datos in validas.items():
        parte = claves.merge(datos, on=CLAVES, validate='one_to_one').sort_values(CLAVES).reset_index(drop=True)
        assert len(parte) == len(claves)
        if reales is None:
            reales = parte.y_real.to_numpy()
        else:
            np.testing.assert_array_equal(parte.y_real.to_numpy(), reales)
        parte['Modelo'] = nombre
        parte['error'] = parte.y_real - parte.y_pred
        parte['error_absoluto'] = parte.error.abs()
        comunes.append(parte[CLAVES + ['Modelo','y_real','y_pred','error','error_absoluto']])
        protocolo = 'Multi-step desde origen fijo' if nombre.startswith('ARIMA') else 'Secuencial semanal, parámetros fijos'
        for marco, universo, destino in [(datos, 'propia', propias), (parte, 'comun', tablas)]:
            destino.append({'Modelo': nombre, **metricas_predicciones(marco),
                'Observaciones_evaluadas': len(marco),
                'Series_evaluadas': len(marco[['sucursal','producto']].drop_duplicates()),
                'Poblacion': universo, 'Protocolo': protocolo})
    comun = pd.concat(comunes, ignore_index=True)
    assert comun.groupby('Modelo').size().nunique() == 1
    propia, tabla = pd.DataFrame(propias), pd.DataFrame(tablas)
    cobertura = pd.DataFrame(cobertura).merge(propia[['Modelo','Observaciones_evaluadas','Series_evaluadas']], on='Modelo')
    cobertura['Observaciones_comunes'] = len(claves)
    cobertura['Series_comunes'] = len(claves[['sucursal','producto']].drop_duplicates())
    cobertura['Observaciones_fuera_comun'] = cobertura.Observaciones_evaluadas - len(claves)
    cobertura['Fraccion_observaciones_retenidas'] = len(claves) / cobertura.Observaciones_evaluadas
    return comun, tabla, propia, cobertura
