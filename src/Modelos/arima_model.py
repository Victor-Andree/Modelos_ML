import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

class ArimaPredictor:
    def __init__(self, dataframe, date_col='fecdoc', target_col='cantidad'):
        self.df = dataframe.copy()
        self.date_col = date_col
        self.target_col = target_col
        
        self.train_data = None
        self.test_data = None
        self.predictions = None
        self.model_fit = None

    def prepare_data(self, test_size=0.2):
        
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        df_semanal = self.df.set_index(self.date_col).resample('W')[self.target_col].sum()
        
        train_len = int(len(df_semanal) * (1 - test_size))
        self.train_data = df_semanal.iloc[:train_len]
        self.test_data = df_semanal.iloc[train_len:]
        
        return self.train_data, self.test_data

    def train_and_evaluate(self, order=(3, 1, 3)):

        # 1. Entrenar
        modelo = ARIMA(self.train_data, order=order)
        self.model_fit = modelo.fit()
        
        # 2. Predecir
        self.predictions = self.model_fit.forecast(steps=len(self.test_data))
        
        # 3. Calcular Métricas
        mae = mean_absolute_error(self.test_data, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.test_data, self.predictions))
        r2 = r2_score(self.test_data, self.predictions)
        
        metricas = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }
        return metricas

    def plot_results(self, title="Predicción de Ventas - ARIMA"):
        """Genera el gráfico comparativo de Real vs Predicción."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_data.index, self.train_data.values, label='Entrenamiento')
        plt.plot(self.test_data.index, self.test_data.values, label='Realidad (Test)', color='blue')
        plt.plot(self.test_data.index, self.predictions, label='Predicción ARIMA', color='red', linestyle='--')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Fecha')
        plt.ylabel('Cantidad Vendida')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()