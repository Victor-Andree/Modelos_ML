import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts import analizar_significancia_modelos as analisis


class SignificanciaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matriz = analisis.cargar_validar()
        cls.f, cls.p, cls.r = analisis.calcular(cls.matriz)

    def test_poblacion_y_consistencia_test(self):
        self.assertEqual(self.matriz.shape, (129, 4))
        self.assertFalse(self.matriz.isna().any().any())

    def test_friedman_y_kendall(self):
        self.assertTrue(self.f.Significativo.iloc[0])
        self.assertAlmostEqual(self.f.Chi2.iloc[0], 57.80930232558126)
        self.assertTrue(0 <= self.f.Kendall_W.iloc[0] <= 1)
        self.assertAlmostEqual(self.f.Kendall_W.iloc[0], self.f.Chi2.iloc[0]/387)

    def test_holm_independiente(self):
        self.assertEqual(len(self.p), 6)
        self.assertTrue(self.p.Series.eq(129).all())
        self.assertTrue(self.p.p_holm.ge(self.p.p_raw).all())
        orden = np.argsort(self.p.p_raw.to_numpy())
        ajustados = np.minimum(1, np.maximum.accumulate(self.p.p_raw.to_numpy()[orden] * np.arange(6, 0, -1)))
        np.testing.assert_allclose(self.p.p_holm.to_numpy()[orden], ajustados)
        np.testing.assert_array_equal(self.p.Significativo_Holm, self.p.p_holm < 0.05)

    def test_rangos(self):
        esperado = [1.9147286821705427, 2.2635658914728682, 2.8372093023255816, 2.9844961240310077]
        np.testing.assert_allclose(self.r.Rango_medio, esperado)

    def test_reproducibilidad_y_archivos(self):
        with contextlib.redirect_stdout(io.StringIO()):
            f, p, r = analisis.ejecutar()
        for actual, esperado in zip((f,p,r),(self.f,self.p,self.r)):
            pd.testing.assert_frame_equal(actual, esperado)
        sal = analisis.ROOT/'resultados/semanal'
        nombres = ['significancia_friedman.csv','significancia_wilcoxon_holm.csv','rangos_mae_por_modelo.csv','manifiesto_significancia.json']
        antes = {n:analisis.sha(sal/n) for n in nombres}
        with contextlib.redirect_stdout(io.StringIO()):
            analisis.ejecutar()
        self.assertEqual(antes,{n:analisis.sha(sal/n) for n in nombres})
        analisis.verificar_integridad()

    def test_rechaza_series_ausentes_y_nan(self):
        with self.assertRaises(AssertionError):
            analisis.calcular(self.matriz.iloc[:-1])
        mala = self.matriz.copy()
        mala.iloc[0,0] = np.nan
        with self.assertRaises(AssertionError):
            analisis.calcular(mala)

    def test_rechaza_semanas_train(self):
        original = pd.read_csv
        def alterado(path, **kwargs):
            datos = original(path, **kwargs)
            if Path(path).name == 'predicciones_comunes.csv':
                datos.loc[0, 'semana'] = '2025-12-28'
            return datos
        with patch.object(analisis.pd, 'read_csv', side_effect=alterado):
            with self.assertRaisesRegex(AssertionError, 'TEST'):
                analisis.cargar_validar()

    def test_integridad_detecta_alteracion(self):
        with tempfile.TemporaryDirectory() as directorio:
            root = Path(directorio)
            (root/'docs').mkdir()
            (root/'original.txt').write_bytes(b'original')
            import json
            (root/'docs/integridad_previa_significancia.json').write_text(json.dumps({'archivos':{'original.txt':analisis.sha(root/'original.txt')}}))
            analisis.verificar_integridad(root)
            (root/'original.txt').write_bytes(b'alterado')
            with self.assertRaisesRegex(AssertionError, 'alterado'):
                analisis.verificar_integridad(root)

    def test_posthoc_condicional_y_ceros(self):
        from types import SimpleNamespace
        with patch.object(analisis, 'friedmanchisquare', return_value=SimpleNamespace(statistic=1.,pvalue=.8)):
            self.assertTrue(analisis.calcular(self.matriz)[1].empty)
        igual = self.matriz.copy()
        igual[analisis.MODELOS[1]] = igual[analisis.MODELOS[0]]
        p = analisis.calcular(igual)[1].iloc[0]
        self.assertEqual(p.Diferencias_cero, 129)
        self.assertEqual(p.Pares_no_cero, 0)
        self.assertEqual(p.p_raw, 1.)


if __name__ == '__main__':
    unittest.main()
