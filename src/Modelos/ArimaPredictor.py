import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools
from src.preprocessing.demanda_semanal import construir_semanal, excluir_semana_corte
from src.evaluacion_semanal import metricas_predicciones

warnings.filterwarnings("ignore")

class ArimaPredictor:
    def __init__(self, dataframe, date_col='fecdoc', target_col='cantidad'):
        # Recibe el dataset limpio con la historia de todas las sucursales
        self.df = dataframe.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.y_real_totales = []
        self.predicciones_totales = []
        self.conteo_series = {}
        self.predicciones = pd.DataFrame()

    def preparar_semanal(self):
        """Misma depuración transaccional y demanda semanal que ML."""
        datos = self.df.rename(columns={self.date_col: 'fecdoc', self.target_col: 'cantidad'})
        self.semanal = construir_semanal(datos)
        self.semanal_evaluable, self.semanas_excluidas = excluir_semana_corte(self.semanal)
        return self.semanal
        
        
    def _determinar_d(self, serie):
        """Aplica la prueba Dickey-Fuller Aumentada (ADF) exigida en la metodología"""
        # 1. Control de seguridad: Si la serie es una línea plana (varianza cero), d = 0
        if serie.var() == 0:
            return 0
            
        d = 0
        try:
            p_value = adfuller(serie.dropna())[1]
            
            # Si p_value > 0.05, la serie no es estacionaria, aplicamos diferenciación
            while p_value > 0.05 and d < 2:
                d += 1
                serie = serie.diff().dropna()
                
                # Volver a verificar que la nueva serie tenga suficientes datos y no sea constante
                if len(serie) > 10 and serie.var() > 0:
                    p_value = adfuller(serie)[1]
                else:
                    break
        except ValueError:
            # Captura cualquier otro error de constancia interna en statsmodels
            return 0
            
        return d


    def _optimizar_pq(self, serie, d):
        """Busca la combinación (p, d, q) que minimice el criterio AIC"""
        p_params = range(0, 4)
        q_params = range(0, 4)
        mejor_aic = float("inf")
        mejor_orden = (0, d, 0)
        
        for p, q in itertools.product(p_params, q_params):
            try:
                modelo = ARIMA(serie, order=(p, d, q))
                resultado = modelo.fit()
                if resultado.aic < mejor_aic:
                    mejor_aic = resultado.aic
                    mejor_orden = (p, d, q)
            except:
                continue
        return mejor_orden

    def entrenar_y_evaluar(self):
        print("Iniciando entrenamiento ARIMA (Iteracion por Sucursal y Producto)...")
        
        self.preparar_semanal()
        grupos = self.semanal.groupby(['sucursal', 'producto'], observed=True)
        detalles = []
        registro = []
        self.predicciones = pd.DataFrame()
        # Reiniciar acumuladores en cada ejecución; potenciales es el producto cartesiano observado.
        self.y_real_totales = []
        self.predicciones_totales = []
        self.conteo_series = {
            'potenciales': self.df['sucursal'].nunique() * self.df['producto'].nunique(),
            'encontradas': grupos.ngroups,
            'descartadas': 0, 'fallidas': 0, 'modeladas': 0
        }
        
        for nombre_grupo, df_grupo in grupos:
            sucursal, producto = nombre_grupo
            estado = {'sucursal': sucursal, 'producto': producto, 'semanas': len(df_grupo)}
            registro.append(estado)
            
            # Calendario completo compartido; conservar frecuencia y distancia al origen.
            df_serie = df_grupo.set_index('semana')['cantidad'].asfreq('W-SUN')
            
            if len(df_serie) < 20:
                estado.update(estado='descartada', motivo='menos_de_20_semanas')
                self.conteo_series['descartadas'] += 1
                continue
                
            # Partición temporal estricta (Evita el Data Leakage)
            # Train: 2024-2025 | Test: 2026
            elegible, _ = excluir_semana_corte(df_grupo)
            serie_evaluable = elegible.set_index('semana')['cantidad']
            train = df_serie[df_serie.index < '2026-01-01']
            test = serie_evaluable[serie_evaluable.index >= '2026-01-01']
            
            estado.update(semanas_train=len(train), semanas_test=len(test))
            if len(train) < 10 or len(test) < 2:
                razones = []
                if len(train) < 10: razones.append('menos_de_10_semanas_train')
                if len(test) < 2: razones.append('menos_de_2_semanas_test')
                estado.update(estado='descartada', motivo=';'.join(razones))
                self.conteo_series['descartadas'] += 1
                continue
                
            try:
                # 1. Determinar parámetro 'd'
                d_optimo = self._determinar_d(train)

                # 2. Determinar parámetros 'p' y 'q' óptimos
                mejor_orden = self._optimizar_pq(train, d_optimo)

                # 3. Entrenar el modelo final para esta sucursal/producto
                modelo = ARIMA(train, order=mejor_orden)
                modelo_fit = modelo.fit()
                
                # 4. Proyectar sobre el periodo de prueba (2026)
                # Pronosticar también la semana excluida y omitirla sólo del scoring.
                # De otro modo el paso 1 (04-ene) se asignaría erróneamente al 11-ene.
                fechas_futuras = pd.date_range(train.index.max() + pd.Timedelta(weeks=1),
                                              test.index.max(), freq='W-SUN')
                valores = modelo_fit.forecast(steps=len(fechas_futuras))
                predicciones = pd.Series(np.asarray(valores, dtype=float), index=fechas_futuras).loc[test.index]
                
                if not np.isfinite(predicciones).all():
                    raise ValueError('Pronóstico ARIMA no finito')

                # Acumular resultados para el cálculo global
                self.y_real_totales.extend(test.values)
                self.predicciones_totales.extend(predicciones.values)
                self.conteo_series['modeladas'] += 1
                detalle = pd.DataFrame({'sucursal': sucursal, 'producto': producto,
                                        'semana': test.index, 'y_real': test.to_numpy(),
                                        'y_pred': predicciones.to_numpy(),
                                        'origen': train.index.max(),
                                        'horizonte_semanas': ((test.index - train.index.max()).days // 7)})
                detalle['error'] = detalle.y_real - detalle.y_pred
                detalle['error_absoluto'] = detalle.error.abs()
                detalles.append(detalle)
                estado.update(estado='modelada', motivo='', orden=str(mejor_orden), aic=float(modelo_fit.aic))
                
            except Exception as exc:
                estado.update(estado='fallida', motivo=f'{type(exc).__name__}: {exc}')
                self.conteo_series['fallidas'] += 1
                continue

        self.registro_series = pd.DataFrame(registro)
        self.predicciones = pd.concat(detalles, ignore_index=True) if detalles else pd.DataFrame()
        print(f'Semanas excluidas por cruce del corte: {len(self.semanas_excluidas)}')
        print(f'Resumen de series ARIMA: {self.conteo_series}')
        if not self.conteo_series['modeladas']:
            raise ValueError('No se modeló ninguna serie; revisar conteo_series.')

        metricas = {'Modelo': 'ARIMA (Optimizado ADF/AIC)',
                    **metricas_predicciones(self.predicciones),
                    'Observaciones_evaluadas': len(self.predicciones),
                    'Series_evaluadas': self.conteo_series['modeladas']}

        print("Entrenamiento completado.")
        return metricas