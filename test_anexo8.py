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

    def test_mase_independiente(self):
        train = self.w.loc[self.w.semana.lt(a.CORTE)].sort_values(a.KEY)
        escala = train.groupby(a.SERIE).cantidad.apply(lambda x:x.diff().abs().mean())
        err = self.hist.assign(ae=(self.hist.y_real-self.hist.y_pred).abs()).groupby(a.SERIE).ae.mean()
        validas = escala.reindex(err.index).gt(0)
        esperado = (err[validas]/escala.reindex(err.index)[validas]).mean()
        row = self.bases.iloc[-1]
        self.assertAlmostEqual(row.MASE,esperado,places=12)
        self.assertEqual(row.Series_MASE_validas,validas.sum())
        conteos = self.hist.groupby(a.SERIE).size()
        self.assertEqual(row.Observaciones_MASE_validas,conteos[validas].sum())

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
