import ast
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.demanda_semanal import (
    CLAVES, NUMERICAS, construir_semanal, agregar_features, particiones_por_semana, excluir_semana_corte)


class SemanalTests(unittest.TestCase):
    def ejemplo(self):
        return pd.DataFrame({'sucursal': [1, 1, 1, 2], 'producto': ['A'] * 4,
                             'fecdoc': ['2025-12-21', '2026-01-01', '2026-01-02', '2026-01-04'],
                             'cantidad': [2., 3., 4., -1.]})

    def test_agregacion_ceros_interiores_y_limites(self):
        original = self.ejemplo()
        semanal = construir_semanal(original)
        self.assertEqual(semanal[semanal.sucursal.eq(1)].cantidad.tolist(), [2., 0., 7.])
        self.assertEqual(len(semanal[semanal.sucursal.eq(2)]), 1)
        self.assertEqual(semanal.cantidad_neta_original.sum(), original.cantidad.sum())
        self.assertEqual(semanal.semana.dt.dayofweek.unique().tolist(), [6])

    def test_nan_no_es_cero(self):
        for columna in ['cantidad', 'fecdoc', 'sucursal', 'producto']:
            df = self.ejemplo(); df[columna] = df[columna].astype(object); df.loc[0, columna] = None
            with self.assertRaises(ValueError): construir_semanal(df)

    def test_features_causales_y_aislamiento(self):
        semanas = pd.date_range('2025-01-05', periods=16, freq='W-SUN')
        df = pd.DataFrame({'sucursal': [1]*16+[2]*16, 'producto': ['A']*32,
                           'semana': list(semanas)*2, 'cantidad': np.arange(32, dtype=float)})
        antes = agregar_features(df)
        df.loc[10:15, 'cantidad'] = 999
        despues = agregar_features(df)
        pd.testing.assert_frame_equal(antes.loc[:10, NUMERICAS], despues.loc[:10, NUMERICAS])
        pd.testing.assert_frame_equal(antes.loc[16:, NUMERICAS], despues.loc[16:, NUMERICAS])
        self.assertEqual(antes.loc[8, 'lag_8'], 0.)
        self.assertEqual(antes.loc[8, 'rolling_mean_4'], 5.5)
        self.assertEqual(antes.loc[8, 'rolling_mean_8'], 3.5)
        self.assertEqual(antes[NUMERICAS].isna().any(axis=1).sum(), 16)

    def test_cv_no_divide_una_semana(self):
        fechas = pd.Series(np.repeat(pd.date_range('2025-01-05', periods=20, freq='W-SUN'), 3))
        folds = particiones_por_semana(fechas)
        self.assertEqual(len(folds), 3)
        for a, b in folds:
            self.assertLess(fechas.iloc[a].max(), fechas.iloc[b].min())
            self.assertFalse(set(fechas.iloc[a]) & set(fechas.iloc[b]))

    def test_encoder_y_predicciones_del_notebook(self):
        nb = json.loads((ROOT/'notebooks/07_Entrenamiento_MachineLearning.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in nb['cells'] if c.get('id') == 'clase-modelos')
        nodo = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
        espacio = dict(globals())
        exec(compile(ast.Module(body=[nodo], type_ignores=[]), '<notebook>', 'exec'), espacio)
        Clase = espacio['MachineLearningPipeline']
        x = pd.DataFrame({c: np.arange(16, dtype=float) for c in NUMERICAS})
        x['sucursal'] = 1; x['producto'] = 'A'
        xt = x.iloc[:2].copy(); xt['producto'] = 'NUEVO'
        y = pd.Series(np.arange(16, dtype=float))
        yt = pd.Series([0.123456789, 1.987654321], index=xt.index)
        claves = xt[CLAVES].copy(); claves['semana'] = pd.date_range('2026-01-04', periods=2, freq='W')
        pipe = Clase(x, y, xt, yt, pd.date_range('2025-01-05', periods=16, freq='W'), claves)
        modelo = pipe._pipeline(LinearRegression()).fit(x, y)
        categorias = modelo.named_steps['preprocesador'].named_transformers_['categorias'].categories_
        self.assertEqual(categorias[1].tolist(), ['A'])
        self.assertEqual(modelo.predict(xt).shape, (2,))
        pipe._registrar_metricas('prueba', [0.1, 2.])
        detalle = pipe.predicciones['prueba']
        np.testing.assert_allclose(detalle.error, detalle.y_real-detalle.y_pred)
        self.assertEqual(pipe.resultados[0]['MAE'], mean_absolute_error(yt, [0.1, 2.]))
        self.assertNotEqual(pipe.resultados[0]['MAE'], round(pipe.resultados[0]['MAE'], 2))
        pd.testing.assert_frame_equal(detalle[CLAVES+['semana']], claves)


class TargetDefinitivoTests(unittest.TestCase):
    def transacciones(self, valores, fechas=None):
        if fechas is None: fechas = ['2025-12-01'] * len(valores)
        return pd.DataFrame({'sucursal':1, 'producto':'A', 'fecdoc':fechas, 'cantidad':valores})

    def test_venta_y_anulacion_neto_cero(self):
        s = construir_semanal(self.transacciones([1,-1]))
        self.assertEqual(s.cantidad_neta_original.tolist(), [0])
        self.assertEqual(s.cantidad.tolist(), [0])

    def test_saldo_menos_dos_truncado_despues_de_sumar(self):
        s = construir_semanal(self.transacciones([1,-3]))
        self.assertEqual(s.cantidad_neta_original.tolist(), [-2])
        self.assertEqual(s.cantidad.tolist(), [0])

    def test_saldo_positivo_no_se_modifica(self):
        s = construir_semanal(self.transacciones([5,-2]))
        self.assertEqual(s.cantidad_neta_original.tolist(), [3])
        self.assertEqual(s.cantidad.tolist(), [3])

    def test_corte_no_contiene_diciembre(self):
        df = self.transacciones([10,20,30], ['2025-12-31','2026-01-01','2026-01-05'])
        s = construir_semanal(df)
        evaluable, fuera = excluir_semana_corte(s)
        self.assertEqual(fuera.semana.tolist(), [pd.Timestamp('2026-01-04')])
        self.assertEqual(fuera.cantidad.tolist(), [30])
        self.assertEqual(evaluable.cantidad.tolist(), [30])
        for semana in evaluable[evaluable.semana.ge('2026-01-01')].semana:
            usadas = pd.to_datetime(df.fecdoc).between(semana-pd.Timedelta(days=6), semana+pd.Timedelta(days=1), inclusive='left')
            self.assertTrue(pd.to_datetime(df.loc[usadas,'fecdoc']).ge('2026-01-01').all())

    def test_lag_no_comprime_semana_excluida(self):
        fechas = pd.date_range('2025-10-26', periods=14, freq='W-SUN')
        s = construir_semanal(self.transacciones(list(range(14)), fechas))
        f, _ = excluir_semana_corte(agregar_features(s))
        valor = f.loc[f.semana.eq('2026-01-11'),'lag_1'].iloc[0]
        self.assertEqual(valor, s.loc[s.semana.eq('2026-01-04'),'cantidad'].iloc[0])

    def test_arima_y_ml_consumen_mismo_target(self):
        from src.Modelos.ArimaPredictor import ArimaPredictor
        df = self.transacciones([1,-3,4,-1], ['2025-12-01','2025-12-02','2025-12-10','2026-01-11'])
        arima = ArimaPredictor(df)
        pd.testing.assert_frame_equal(arima.preparar_semanal(), construir_semanal(df))
        self.assertTrue(arima.semanal.cantidad.ge(0).all())
        esperado, _ = excluir_semana_corte(construir_semanal(df))
        pd.testing.assert_frame_equal(arima.semanal_evaluable, esperado)

    def test_arima_respeta_horizonte_al_saltar_cruce(self):
        from unittest.mock import patch
        from src.Modelos.ArimaPredictor import ArimaPredictor
        fechas = pd.date_range('2025-08-03', '2026-02-01', freq='W-SUN')
        df = self.transacciones([1]*len(fechas), fechas)
        class FakeARIMA:
            def __init__(self, train, order):
                assert train.index.max() == pd.Timestamp('2025-12-28')
                assert train.min() >= 0
            def fit(self): return self
            def forecast(self, steps): return np.arange(1, steps+1, dtype=float)
        modelo = ArimaPredictor(df)
        with patch('src.Modelos.ArimaPredictor.ARIMA', FakeARIMA), \
             patch.object(modelo, '_determinar_d', return_value=0), \
             patch.object(modelo, '_optimizar_pq', return_value=(0,0,0)):
            modelo.entrenar_y_evaluar()
        self.assertEqual(modelo.predicciones.semana.iloc[0], pd.Timestamp('2026-01-11'))
        self.assertEqual(modelo.predicciones.y_pred.iloc[0], 2)
        self.assertEqual(modelo.predicciones.horizonte_semanas.iloc[0], 2)


if __name__ == '__main__':
    unittest.main()
