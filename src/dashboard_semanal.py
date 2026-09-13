"""Inferencia semanal compartida con EDA 07."""
import pandas as pd
from src.preprocessing.demanda_semanal import CLAVES, NUMERICAS, agregar_features

def entrada_pronostico(semanal, sucursal, producto, fecha_completa):
    cobertura = pd.Timestamp(fecha_completa).normalize()
    domingo = cobertura - pd.Timedelta(days=(cobertura.dayofweek + 1) % 7)
    historia = semanal.loc[(semanal.sucursal == sucursal) &
        (semanal.producto == producto) & (semanal.semana <= domingo)].copy()
    esperadas = pd.date_range(end=domingo, periods=8, freq='W-SUN')
    if not set(esperadas).issubset(set(historia.semana)):
        raise ValueError('Se requieren ocho semanas consecutivas hasta el último domingo cerrado. No se imputan ceros después del fin de una serie.')
    siguiente = domingo + pd.Timedelta(days=7)
    nueva = pd.DataFrame([{'sucursal': sucursal, 'producto': producto,
                          'semana': siguiente, 'cantidad': float('nan')}])
    features = agregar_features(pd.concat([historia, nueva], ignore_index=True))
    entrada = features.loc[features.semana.eq(siguiente), CLAVES + NUMERICAS]
    if entrada.isna().any().any():
        raise ValueError('Historia insuficiente para calcular los predictores.')
    return siguiente, entrada
