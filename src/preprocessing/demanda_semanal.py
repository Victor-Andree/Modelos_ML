import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

CLAVES = ['sucursal', 'producto']
LAGS = [1, 2, 4, 8]
NUMERICAS = ['lag_1', 'lag_2', 'lag_4', 'lag_8', 'rolling_mean_4',
             'rolling_mean_8', 'mes', 'semana_del_anio', 'trimestre']


def construir_semanal(df):
    """Suma neta ERP; conserva el neto y trunca sólo el saldo semanal del target.

    Ceros sólo en huecos interiores sin registros; NaN de origen es un error.
    No convierte cajas/unidades ni elimina transacciones negativas.
    """
    datos = df[CLAVES + ['fecdoc', 'cantidad']].copy()
    if datos.empty or datos.isna().any().any():
        raise ValueError('Dataset vacío o campos requeridos desconocidos; no imputar NaN como cero.')
    datos['fecdoc'] = pd.to_datetime(datos['fecdoc'], errors='raise')
    datos['cantidad'] = pd.to_numeric(datos['cantidad'], errors='raise')
    if datos['fecdoc'].isna().any() or not np.isfinite(datos['cantidad']).all():
        raise ValueError('Fechas o cantidades no válidas.')
    partes = []
    for (sucursal, producto), grupo in datos.groupby(CLAVES, observed=True):
        resample = grupo.set_index('fecdoc')['cantidad'].resample('W-SUN')
        cantidad = resample.sum(min_count=1)
        cantidad.loc[resample.size().eq(0)] = 0
        parte = cantidad.rename('cantidad').rename_axis('semana').reset_index()
        parte['sucursal'], parte['producto'] = sucursal, producto
        partes.append(parte)
    semanal = pd.concat(partes, ignore_index=True)[CLAVES + ['semana', 'cantidad']]
    semanal['cantidad_neta_original'] = semanal['cantidad']
    semanal['cantidad'] = semanal['cantidad_neta_original'].clip(lower=0)
    return semanal[CLAVES + ['semana', 'cantidad_neta_original', 'cantidad']].sort_values(
        CLAVES + ['semana']).reset_index(drop=True)


def agregar_features(semanal):
    datos = semanal.sort_values(CLAVES + ['semana']).copy()
    grupo = datos.groupby(CLAVES, observed=True)['cantidad']
    for lag in LAGS:
        datos[f'lag_{lag}'] = grupo.shift(lag)
    for ventana in [4, 8]:
        datos[f'rolling_mean_{ventana}'] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=ventana).mean())
    datos['mes'] = datos['semana'].dt.month
    datos['semana_del_anio'] = datos['semana'].dt.isocalendar().week.astype(int)
    datos['trimestre'] = datos['semana'].dt.quarter
    return datos


def particiones_por_semana(fechas, n_splits=3):
    """TimeSeriesSplit sobre semanas únicas: ninguna semana aparece en ambos bloques."""
    fechas = pd.Series(fechas).reset_index(drop=True)
    semanas = np.sort(fechas.unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    particiones = []
    for train, validacion in tscv.split(semanas):
        a = np.flatnonzero(fechas.isin(semanas[train]).to_numpy())
        b = np.flatnonzero(fechas.isin(semanas[validacion]).to_numpy())
        assert fechas.iloc[a].max() < fechas.iloc[b].min()
        particiones.append((a, b))
    return particiones


def resumir_intermitencia(semanal, target="cantidad"):
    return semanal.groupby(CLAVES, observed=True).agg(
        semanas_totales=(target, 'size'),
        semanas_demanda_cero=(target, lambda s: int(s.eq(0).sum())),
        porcentaje_demanda_cero=(target, lambda s: s.eq(0).mean() * 100),
        demanda_promedio_semanal=(target, 'mean'),
        inicio=('semana', 'min'), fin=('semana', 'max')
    ).reset_index()


def excluir_semana_corte(semanal, fecha_corte='2026-01-01'):
    """Excluye del scoring la semana que cruza el corte, sin comprimir el calendario.

    Calcular lags sobre el calendario completo ANTES de aplicar esta exclusión.
    El intervalo de W-SUN es [lunes 00:00, lunes siguiente 00:00).
    """
    corte = pd.Timestamp(fecha_corte)
    inicio = semanal['semana'] - pd.Timedelta(days=6)
    fin_exclusivo = semanal['semana'] + pd.Timedelta(days=1)
    cruza = inicio.lt(corte) & fin_exclusivo.gt(corte)
    evaluable, excluidas = semanal.loc[~cruza].copy(), semanal.loc[cruza].copy()
    test = evaluable[evaluable.semana.ge(corte)]
    assert (test.semana - pd.Timedelta(days=6)).ge(corte).all()
    train = evaluable[evaluable.semana.lt(corte)]
    assert (train.semana + pd.Timedelta(days=1)).le(corte).all()
    return evaluable, excluidas


def auditar_target(df):
    """Conteos por condición y producto, incluyendo el total sin mezclar unidades."""
    if df[['cantidad', 'unidades', 'producto']].isna().any().any():
        raise ValueError('Auditoría requiere cantidad, unidades y producto conocidos.')
    condiciones = {
        'cantidad_cero_unidades_positivas': df.cantidad.eq(0) & df.unidades.gt(0),
        'cantidad_positiva_unidades_positivas': df.cantidad.gt(0) & df.unidades.gt(0),
        'cantidad_negativa': df.cantidad.lt(0)
    }
    filas = []
    for condicion, mascara in condiciones.items():
        grupo = df.loc[mascara]
        filas.append({'condicion': condicion, 'producto': '__TOTAL__',
                      'registros': len(grupo), 'suma_cantidad': grupo.cantidad.sum(),
                      'productos_afectados': grupo.producto.nunique()})
        for producto, parte in grupo.groupby('producto', observed=True):
            filas.append({'condicion': condicion, 'producto': producto,
                          'registros': len(parte), 'suma_cantidad': parte.cantidad.sum(),
                          'productos_afectados': 1})
    return pd.DataFrame(filas)
