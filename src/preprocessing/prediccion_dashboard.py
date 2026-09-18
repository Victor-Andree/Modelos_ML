"""Entrada semanal de una sola serie, sin observar la semana pronosticada."""
import pandas as pd
from .demanda_semanal import CLAVES, NUMERICAS, agregar_features


def preparar_entrada(semanal, sucursal, producto, origen):
    origen = pd.Timestamp(origen).normalize()
    if origen.dayofweek != 6:
        raise ValueError('El origen debe ser un domingo cerrado.')
    historia = semanal.loc[
        semanal.sucursal.eq(sucursal) & semanal.producto.eq(producto)
        & semanal.semana.le(origen)
    ].sort_values('semana').copy()
    esperadas = pd.date_range(end=origen, periods=8, freq='W-SUN')
    if not pd.DatetimeIndex(historia.semana.tail(8)).equals(esperadas):
        raise ValueError('Se requieren ocho semanas consecutivas hasta el origen; no se imputan ceros fuera de la cobertura de la serie.')
    futura = historia.iloc[[-1]].copy()
    futura['semana'] = origen + pd.Timedelta(days=7)
    futura['cantidad'] = float('nan')
    features = agregar_features(pd.concat([historia, futura], ignore_index=True))
    entrada = features.iloc[[-1]][CLAVES + NUMERICAS]
    if entrada.isna().any().any():
        raise ValueError('Historia insuficiente para construir las variables.')
    return entrada, futura.semana.iloc[0]
