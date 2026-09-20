import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.extension_intermitente import features_intermitentes,predecir_serie,umbral_train,escalas_train,metricas,ejecutar,NUMERICAS,EXTRAS
ROOT=Path(__file__).resolve().parent

class CausalidadExtensionTests(unittest.TestCase):
    def weekly(self,values):
        return pd.DataFrame({'sucursal':1,'producto':'A','semana':pd.date_range('2025-01-05',periods=len(values),freq='W-SUN'),'cantidad':values})

    def test_feature_actual_y_futuro_no_cambian_pasado(self):
        d=self.weekly([0,1,0,2,0,0,4,0,0,3,0,0,7,0,0,1])
        before=features_intermitentes(d)
        d.loc[10:,'cantidad']=999
        after=features_intermitentes(d)
        pd.testing.assert_frame_equal(before.loc[:10,NUMERICAS+EXTRAS],after.loc[:10,NUMERICAS+EXTRAS])

    def test_distancia_y_magnitudes_solo_pasadas(self):
        f=features_intermitentes(self.weekly([0,0,4,0,2,0]))
        self.assertEqual(f.semanas_desde_ultima_demanda.tolist(),[1,2,3,1,2,1])
        self.assertEqual(f.demanda_no_cero_lag_1.tolist(),[0,0,0,4,4,2])
        self.assertEqual(f.demanda_no_cero_lag_2.tolist(),[0,0,0,0,0,4])
        self.assertEqual(f.media_demanda_positiva_historica.tolist(),[0,0,0,4,4,3])

    def test_conteos_rolling_shift(self):
        f=features_intermitentes(self.weekly([1,0,2,0,3,0,4,0,999]))
        self.assertEqual(f.loc[8,'numero_eventos_positivos_ultimas_4'],2)
        self.assertEqual(f.loc[8,'numero_eventos_positivos_ultimas_8'],4)
        self.assertEqual(f.loc[8,'tasa_demanda_positiva_ultimas_8'],.5)

    def test_aislamiento_entre_series(self):
        a=self.weekly([0,1,0,1]*3);b=a.copy();b.sucursal=2;b.cantidad=100
        conjunto=features_intermitentes(pd.concat([a,b]))
        pd.testing.assert_frame_equal(features_intermitentes(a),conjunto.loc[conjunto.sucursal.eq(1)].reset_index(drop=True))

    def test_croston_sba_tsb_a_mano(self):
        y=[4,0,2,0]
        np.testing.assert_allclose(predecir_serie(y,'Croston',.5,.5),[0,4,4,2])
        np.testing.assert_allclose(predecir_serie(y,'SBA',.5,.5),[0,3,3,1.5])
        np.testing.assert_allclose(predecir_serie(y,'TSB',.5,.5),[0,4,2,2.25])

    def test_baselines_no_observan_objetivo(self):
        a=np.array([0,0,6,0,1,0],dtype=float);b=a.copy();b[3:]=999
        for nombre in ['Cero','Naive_ultimo','Croston','SBA','TSB']:
            np.testing.assert_array_equal(predecir_serie(a,nombre)[:4],predecir_serie(b,nombre)[:4])
        np.testing.assert_array_equal(predecir_serie([0,0,0],'TSB'),[0,0,0])

    def test_umbral_no_fijado_automaticamente_en_05(self):
        d=pd.DataFrame({'semana':pd.to_datetime(['2025-01-05','2025-01-12']),'y_real':[0,1],'probabilidad':[.1,.2],'magnitud':[1,1]})
        mejor,_=umbral_train(d)
        self.assertEqual(mejor['umbral'],.15)
        d.loc[1,'semana']=pd.Timestamp('2026-01-11')
        with self.assertRaises(AssertionError):umbral_train(d)

    def test_escala_no_acepta_test_y_constantes_son_indefinidas(self):
        s=self.weekly([0,0,0]);esc=escalas_train(s)
        d=s.rename(columns={'cantidad':'y_real'});d['y_pred']=0
        met=metricas(d,esc)
        self.assertTrue(np.isnan(met['MASE']) and np.isnan(met['RMSSE']) and np.isnan(met['R2']))
        self.assertEqual(met['observaciones'],3)
        s.loc[2,'semana']=pd.Timestamp('2026-01-11')
        with self.assertRaises(AssertionError):escalas_train(s)

    def test_bloqueo_repeticion_test_antes_de_fit(self):
        with patch('src.extension_intermitente.seleccionar',side_effect=AssertionError('No debe seleccionar otra vez')):
            with self.assertRaises(RuntimeError):ejecutar(ROOT)

    def test_integridad_y_metricas_reproducibles(self):
        from scripts.validar_extension_intermitente import validar
        validar()

if __name__=='__main__':unittest.main()
