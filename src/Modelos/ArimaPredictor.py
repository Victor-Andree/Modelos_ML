import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

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
        
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        grupos = self.df.groupby(['sucursal', 'producto'], observed=True)
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
            
            # Resampleo semanal para estandarizar la serie temporal
            df_serie = df_grupo.set_index(self.date_col).resample('W')[self.target_col].sum().fillna(0)
            
            if len(df_serie) < 20:
                self.conteo_series['descartadas'] += 1
                continue
                
            # Partición temporal estricta (Evita el Data Leakage)
            # Train: 2024-2025 | Test: 2026
            train = df_serie[df_serie.index < '2026-01-01']
            test = df_serie[df_serie.index >= '2026-01-01']
            
            if len(train) < 10 or len(test) < 2:
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
                predicciones = modelo_fit.forecast(steps=len(test))
                
                if not np.isfinite(predicciones).all():
                    raise ValueError('Pronóstico ARIMA no finito')

                # Acumular resultados para el cálculo global
                self.y_real_totales.extend(test.values)
                self.predicciones_totales.extend(predicciones.values)
                self.conteo_series['modeladas'] += 1
                
            except Exception:
                self.conteo_series['fallidas'] += 1
                continue

        print(f'Resumen de series ARIMA: {self.conteo_series}')
        if not self.conteo_series['modeladas']:
            raise ValueError('No se modeló ninguna serie; revisar conteo_series.')

        # 5. Calcular métricas globales consolidadadas (R2, MAE, RMSE)
        mae = mean_absolute_error(self.y_real_totales, self.predicciones_totales)
        rmse = np.sqrt(mean_squared_error(self.y_real_totales, self.predicciones_totales))
        r2 = r2_score(self.y_real_totales, self.predicciones_totales)
        
        metricas = {
            "Modelo": "ARIMA (Optimizado ADF/AIC)",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
        
        print("Entrenamiento completado.")
        return metricas