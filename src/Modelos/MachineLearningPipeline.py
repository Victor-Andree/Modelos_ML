import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.preprocessing.demanda_semanal import CLAVES, NUMERICAS, particiones_por_semana
from src.evaluacion_semanal import metricas_predicciones

class MachineLearningPipeline:
    def __init__(self, X_train, y_train, X_test, y_test, fechas_train, claves_test):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.claves_test = claves_test.copy()
        self.resultados, self.predicciones, self.modelos = [], {}, {}
        self.best_params_, self.best_score_, self.cv_results_ = {}, {}, {}
        self.cv = particiones_por_semana(fechas_train, n_splits=3)

    def _pipeline(self, modelo):
        preprocesador = ColumnTransformer([
            ('categorias', OneHotEncoder(handle_unknown='ignore', drop='first',
                                          sparse_output=False), CLAVES),
            ('numericas', 'passthrough', NUMERICAS)
        ])
        return Pipeline([('preprocesador', preprocesador), ('modelo', modelo)])

    def _registrar_metricas(self, nombre_modelo, predicciones):
        detalle = self.claves_test.copy()
        detalle['y_real'] = np.asarray(self.y_test, dtype=float)
        detalle['y_pred'] = np.asarray(predicciones, dtype=float)
        if not np.isfinite(detalle[['y_real', 'y_pred']]).all().all():
            raise ValueError('Valores no finitos en la evaluación')
        detalle['error'] = detalle.y_real - detalle.y_pred
        detalle['error_absoluto'] = detalle.error.abs()
        self.predicciones[nombre_modelo] = detalle
        self.resultados.append({
            'Modelo': nombre_modelo,
            **metricas_predicciones(detalle),
            'Observaciones_evaluadas': len(detalle),
            'Series_evaluadas': len(detalle[CLAVES].drop_duplicates()),
            'Best Params': self.best_params_.get(nombre_modelo, {}),
            'CV neg_MSE': self.best_score_.get(nombre_modelo, np.nan)
        })

    def ejecutar_regresion_lineal(self):
        nombre = 'Regresión Lineal'
        modelo = self._pipeline(LinearRegression())
        modelo.fit(self.X_train, self.y_train)
        self.modelos[nombre] = modelo
        self._registrar_metricas(nombre, modelo.predict(self.X_test))

    def _buscar(self, nombre, estimador, rejilla):
        grid = GridSearchCV(self._pipeline(estimador),
                            {f'modelo__{k}': v for k, v in rejilla.items()},
                            cv=self.cv, scoring='neg_mean_squared_error',
                            n_jobs=1, error_score='raise', refit=True)
        grid.fit(self.X_train, self.y_train)
        self.modelos[nombre] = grid.best_estimator_
        self.best_params_[nombre] = {k.removeprefix('modelo__'): v for k, v in grid.best_params_.items()}
        self.best_score_[nombre] = float(grid.best_score_)
        self.cv_results_[nombre] = pd.DataFrame(grid.cv_results_)
        print(nombre, 'best_params_:', self.best_params_[nombre])
        print(nombre, 'best_score_ (MSE negativo):', grid.best_score_)
        self._registrar_metricas(nombre, grid.predict(self.X_test))

    def ejecutar_random_forest(self):
        self._buscar('Random Forest', RandomForestRegressor(random_state=42, n_jobs=-1), {
            'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_leaf': [1, 2]})

    def ejecutar_xgboost(self):
        self._buscar('XGBoost', XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1), {
            'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]})

    def obtener_tabla_resultados(self):
        return pd.DataFrame(self.resultados).sort_values('RMSE').reset_index(drop=True)

