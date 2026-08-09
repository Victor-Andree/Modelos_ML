import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RandomForestPredictor:
    def __init__(self, n_estimators=100):
        self.modelo = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.modelo.fit(X_train, y_train)
        predicciones = self.modelo.predict(X_test)
        
        return {
            "Modelo": "Random Forest",
            "MAE": round(mean_absolute_error(y_test, predicciones), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, predicciones)), 2),
            "R2": round(r2_score(y_test, predicciones), 4)
        }