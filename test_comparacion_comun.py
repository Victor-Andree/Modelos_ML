import unittest
import numpy as np
import pandas as pd
from src.preprocessing.clean_data import depurar_demanda
from src.evaluacion_semanal import comparar, ARCHIVOS

class ComunTests(unittest.TestCase):
    def test_limpieza_no_muta_y_audita_nulos_negativos(self):
        raw = pd.DataFrame({'sucursal':[1,1,2,2], 'producto':['A','A','B','B'],
                            'cantidad':[5.,-2.,np.nan,0.]})
        copia = raw.copy(deep=True)
        limpio, a = depurar_demanda(raw)
        pd.testing.assert_frame_equal(raw, copia)
        self.assertEqual(limpio.cantidad.tolist(), [5.,0.])
        self.assertEqual(a['registros_negativos_eliminados'],1)
        self.assertEqual(a['registros_nulos_eliminados'],1)
        self.assertEqual(a['suma_cantidades_negativas'],-2)
        self.assertEqual(a['productos_afectados'],['A','B'])

    def datos(self):
        d = pd.DataFrame({'sucursal':[1]*4, 'producto':['A']*4,
            'semana':pd.date_range('2026-01-11',periods=4,freq='W-SUN'),
            'y_real':[0.,1.,2.,3.], 'y_pred':[0.5,1.,1.,3.]})
        return {n:d.copy() for n in ARCHIVOS}

    def test_interseccion_exacta_y_metricas(self):
        d = self.datos()
        d['XGBoost'] = d['XGBoost'].iloc[1:]
        d['Random Forest'] = d['Random Forest'].iloc[:-1]
        comun, tabla, propias, cobertura = comparar(d)
        self.assertEqual(len(comun),8)
        self.assertEqual(tabla.Observaciones_evaluadas.tolist(),[2]*4)
        self.assertEqual(tabla.Series_evaluadas.tolist(),[1]*4)
        np.testing.assert_allclose(tabla.MAE,0.5)
        np.testing.assert_allclose(tabla.RMSE,np.sqrt(0.5))
        np.testing.assert_allclose(tabla.R2,-1.)
        self.assertEqual(propias.Observaciones_evaluadas.tolist(),[4,3,3,4])

    def test_duplicados_rechazados(self):
        d = self.datos(); d['XGBoost'] = pd.concat([d['XGBoost'],d['XGBoost'].iloc[:1]])
        with self.assertRaises(ValueError): comparar(d)

    def test_targets_distintos_rechazados(self):
        d = self.datos(); d['XGBoost'].loc[0,'y_real']=9
        with self.assertRaises(AssertionError): comparar(d)

    def test_prediccion_no_finita_no_entra_a_interseccion(self):
        d = self.datos(); d['XGBoost'].loc[0,'y_pred']=np.nan
        comun, tabla, propias, cobertura = comparar(d)
        self.assertEqual(tabla.Observaciones_evaluadas.tolist(),[3]*4)
        self.assertEqual(cobertura.set_index('Modelo').loc['XGBoost','Predicciones_no_finitas'],1)

if __name__ == '__main__': unittest.main()
