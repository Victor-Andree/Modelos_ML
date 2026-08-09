import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class XGBoostPredictor:
    def __init__(self, dataframe, date_col='fecdoc', target_col='cantidad'):
        self.df = dataframe.copy()
        self.date_col = date_col
        self.target_col = target_col
        
        self.modelo = XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42,
            objective='reg:squarederror'
        )

    def prepare_data(self, test_size=0.2, lags=3):
        
        # 1. Agrupar por semana igual que ARIMA
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        df_semanal = self.df.set_index(self.date_col).resample('W')[self.target_col].sum().reset_index()
        
        # 2. Crear las variables predictoras (Lags)
        for i in range(1, lags + 1):
            df_semanal[f'lag_{i}'] = df_semanal[self.target_col].shift(i)
            
        # 3. Agregar el mes como variable para que entienda las estaciones
        df_semanal['mes'] = df_semanal[self.date_col].dt.month
        
        # 4. Eliminar los nulos generados por los lags al inicio
        df_semanal.dropna(inplace=True)
        
        # 5. Dividir en Train / Test cronológicamente
        train_len = int(len(df_semanal) * (1 - test_size))
        
        # Separar variables predictoras (X) de lo que queremos predecir (y)
        self.X = df_semanal.drop(columns=[self.date_col, self.target_col])
        self.y = df_semanal[self.target_col]
        self.fechas = df_semanal[self.date_col]
        
        self.X_train, self.X_test = self.X.iloc[:train_len], self.X.iloc[train_len:]
        self.y_train, self.y_test = self.y.iloc[:train_len], self.y.iloc[train_len:]
        
        self.fechas_train = self.fechas.iloc[:train_len]
        self.fechas_test = self.fechas.iloc[train_len:]

    def train_and_evaluate(self):
        
        # Entrenar
        self.modelo.fit(self.X_train, self.y_train)
        
        # Predecir
        self.predictions = self.modelo.predict(self.X_test)
        
        # Métricas
        mae = mean_absolute_error(self.y_test, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.predictions))
        r2 = r2_score(self.y_test, self.predictions)
        
        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }

    def plot_results(self, title="Predicción de Ventas - XGBoost"):

        plt.figure(figsize=(12, 6))
        plt.plot(self.fechas_train, self.y_train, label='Entrenamiento')
        plt.plot(self.fechas_test, self.y_test, label='Realidad (Test)', color='blue')
        plt.plot(self.fechas_test, self.predictions, label='Predicción XGBoost', color='green', linestyle='--')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Fecha')
        plt.ylabel('Cantidad Vendida')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()