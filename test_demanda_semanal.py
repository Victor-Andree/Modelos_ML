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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.demanda_semanal import (
    CLAVES, NUMERICAS, construir_semanal, agregar_features, particiones_por_semana)


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
        self.assertEqual(semanal.cantidad.sum(), original.cantidad.sum())
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


if __name__ == '__main__':
    unittest.main()
