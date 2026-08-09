import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RegresionLinealPredictor:
    def __init__(self):
        self.modelo = LinearRegression()

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.modelo.fit(X_train, y_train)
        predicciones = self.modelo.predict(X_test)
        
        return {
            "Modelo": "Regresión Lineal",
            "MAE": round(mean_absolute_error(y_test, predicciones), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, predicciones)), 2),
            "R2": round(r2_score(y_test, predicciones), 4)
        }