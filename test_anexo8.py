import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts import generar_anexo8 as a


class Anexo8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelos,cls.bases,cls.propia,cls.hist = a.preparar()
        cls.w = a.leer(a.ROOT/'resultados/semanal/dataset_semanal.csv')

    def test_media_train_completa_con_ceros(self):
        train = self.w.loc[self.w.semana.lt(a.CORTE)]
        medias = train.groupby(a.SERIE).cantidad.mean()
        h = self.hist.set_index(a.SERIE)
        np.testing.assert_array_equal(h.media_train,medias.reindex(h.index))
        self.assertTrue(self.hist.groupby(a.SERIE).y_pred.nunique().eq(1).all())

    def test_test_no_influye_en_media(self):
        w = self.w.copy()
        w.loc[w.semana.ge(a.CORTE),'cantidad'] = np.arange(w.semana.ge(a.CORTE).sum())+10000
        ref = self.hist.copy()
        ref['y_real'] = -12345
        original = a.media_historica(self.w,self.hist)
        alterada = a.media_historica(w,ref)
        np.testing.assert_array_equal(original.media_train,alterada.media_train)
        np.testing.assert_array_equal(original.y_pred,alterada.y_pred)

    def test_claves_y_real_cuatro_modelos(self):
        p = a.leer(a.ROOT/'resultados/semanal/predicciones_comunes.csv')
        for _,d in p.groupby('Modelo'):
            a.validar_claves(self.hist,d)
        self.assertFalse(self.hist.duplicated(a.KEY).any())

    def test_sin_historia_se_detiene(self):
        serie = self.hist.iloc[0]
        w = self.w.loc[~((self.w.sucursal==serie.sucursal)&(self.w.producto==serie.producto)&self.w.semana.lt(a.CORTE))]
        with self.assertRaisesRegex(AssertionError,'sin historial'):
            a.media_historica(w,self.hist)

    def test_metricas_y_dispersion(self):
        for row in self.modelos.itertuples():
            np.testing.assert_allclose([row.MAE,row.RMSE,row.R2],a.ESPERADAS[row.Modelo],rtol=0,atol=1e-12)
        por = a.leer(a.ROOT/'resultados/semanal/metricas_por_serie.csv').groupby('Modelo').MAE
        for row in self.modelos.itertuples():
            self.assertAlmostEqual(row.DE_MAE_por_serie,por.std(ddof=1)[row.Modelo],places=12)
            self.assertAlmostEqual(row.MAE_medio_por_serie,por.mean()[row.Modelo],places=12)
        for d in [self.modelos,self.bases]:
            self.assertTrue(np.isfinite(d[['MAE','RMSE','R2']]).all().all())

    def test_controles_lineas_base(self):
        esperadas = [[0.2009291521,0.6854214435,-0.0940142078],
                     [0.2427409988,0.7225422919,-0.2157215981],
                     [0.2207179532,0.5474267690,0.3021539114]]
        np.testing.assert_allclose(self.bases[['MAE','RMSE','R2']],esperadas,rtol=0,atol=1e-10)

    def test_salida_limitada_y_reproducible(self):
        ruta = a.ROOT/'resultados/anexo8'
        with contextlib.redirect_stdout(io.StringIO()):
            a.ejecutar()
        antes = {p.name:p.read_bytes() for p in ruta.glob('*.csv')}
        with contextlib.redirect_stdout(io.StringIO()):
            a.ejecutar()
        self.assertEqual(antes,{p.name:p.read_bytes() for p in ruta.glob('*.csv')})
        columnas = ['Modelo','MAE','RMSE','R2','MAE_medio_por_serie','DE_MAE_por_serie','Observaciones','Series']
        for nombre in ['modelos_poblacion_comun.csv','lineas_base_poblacion_comun.csv']:
            self.assertEqual(list(a.leer(ruta/nombre)),columnas)
        for p in list(ruta.iterdir())+[a.ROOT/'scripts/generar_anexo8.py']:
            texto = p.read_text(encoding='utf-8').upper()
            for termino in ['MA'+'SE','RM'+'SSE']:
                self.assertNotIn(termino,texto)
        self.assertEqual(set(p.name for p in ruta.iterdir()),
            {'README.md','modelos_poblacion_comun.csv','lineas_base_poblacion_comun.csv',
             'modelos_poblacion_propia.csv','predicciones_media_historica.csv','tabla_anexo8.csv'})

    def test_no_fit_y_archivos_previos_intactos(self):
        from sklearn.linear_model import LinearRegression,LogisticRegression
        from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
        from xgboost import XGBRegressor,XGBClassifier
        from statsmodels.tsa.arima.model import ARIMA
        antes = a.huellas(a.ROOT)
        with contextlib.ExitStack() as stack:
            for modelo in [LinearRegression,LogisticRegression,RandomForestRegressor,RandomForestClassifier,XGBRegressor,XGBClassifier,ARIMA]:
                stack.enter_context(patch.object(modelo,'fit',side_effect=AssertionError('Entrenamiento prohibido')))
            with contextlib.redirect_stdout(io.StringIO()):
                a.ejecutar()
        self.assertEqual(antes,a.huellas(a.ROOT))
        self.assertEqual((a.ROOT/'resultados/anexo8/modelos_poblacion_propia.csv').read_bytes(),
                         (a.ROOT/'resultados/semanal/metricas_cobertura_propia.csv').read_bytes())


if __name__=='__main__':
    unittest.main()
