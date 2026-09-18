import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from src.analisis_intermitencia import clasificar, metricas, cruzar, calcular, intervalos_ceros, conteo_minimos, MODELOS
ROOT=Path(__file__).resolve().parent

class ClasificacionTests(unittest.TestCase):
    def serie(self, valores):
        return pd.DataFrame({'sucursal':1,'producto':'A','semana':pd.date_range('2025-01-05',periods=len(valores),freq='W-SUN'),'cantidad':valores})

    def test_cuatro_patrones_y_excepciones(self):
        ejemplos=[([1,1,1,1],'suave'),([1,1,1,10],'erratica'),([0,1,0,1],'intermitente'),([0,1,0,10],'lumpy'),([0,0,0,1],'positivas_insuficientes'),([0,0,0,0],'sin_demanda_positiva')]
        for valores,patron in ejemplos:
            with self.subTest(patron=patron):
                fila=clasificar(self.serie(valores)).iloc[0]
                self.assertEqual(fila.clasificacion_demanda,patron)
        sin=clasificar(self.serie([0,0,0,0])).iloc[0]
        self.assertTrue(np.isnan(sin.ADI) and np.isnan(sin.CV2))

    def test_cv_usa_solo_positivos_y_ddof_1(self):
        fila=clasificar(self.serie([0,1,0,3])).iloc[0]
        self.assertEqual(fila.ADI,2)
        self.assertAlmostEqual(fila.CV2,.5)
        self.assertEqual(fila.porcentaje_ceros,50)

    def test_futuro_y_semana_cruzada_no_cambian_clase(self):
        d=self.serie([0,1,0,3]);antes=clasificar(d)
        futuro=pd.DataFrame({'sucursal':[1,1],'producto':['A','A'],'semana':pd.to_datetime(['2026-01-04','2026-01-11']),'cantidad':[999,888]})
        pd.testing.assert_frame_equal(antes,clasificar(pd.concat([d,futuro])))

    def test_sin_historia_train(self):
        d=self.serie([1,2]);d['semana']=pd.to_datetime(['2026-01-11','2026-01-18'])
        self.assertEqual(clasificar(d).iloc[0].clasificacion_demanda,'sin_historia_train')

    def test_r2_constante_y_negativo(self):
        d=pd.DataFrame({'sucursal':[1,1],'producto':['A','A'],'y_real':[0.,0.],'y_pred':[1.,1.]})
        self.assertTrue(np.isnan(metricas(d)['R2']))
        d.y_real=[1.,2.];d.y_pred=[10.,10.]
        self.assertLess(metricas(d)['R2'],0)

    def test_intervalos_repetidos_no_pierden_series(self):
        d=pd.DataFrame({'porcentaje_ceros':[0,50,100,100,100]})
        result,bordes=intervalos_ceros(d)
        self.assertEqual(len(result),5)
        self.assertTrue(result.intervalo_ceros.notna().all())
        self.assertLess(len(bordes),4)

class IntegridadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s=ROOT/'resultados/semanal'
        cls.p=pd.read_csv(s/'predicciones_comunes.csv',parse_dates=['semana'],float_precision='round_trip')
        cls.s=pd.read_csv(s/'dataset_semanal.csv',parse_dates=['semana'])
        cls.cov=pd.read_csv(s/'resumen_cobertura.csv')

    def test_poblacion_real_y_clasificacion_unica(self):
        c=clasificar(self.s);unido=cruzar(self.p,c)
        self.assertEqual(len(unido),3444*4)
        self.assertEqual(len(unido[['sucursal','producto']].drop_duplicates()),129)
        self.assertEqual(set(unido.Modelo),set(MODELOS))

    def test_clasificacion_duplicada_o_faltante_rechazada(self):
        c=clasificar(self.s)
        comunes=self.p[['sucursal','producto']].drop_duplicates()
        c=c.merge(comunes,on=['sucursal','producto'])
        with self.assertRaises(AssertionError):cruzar(self.p,pd.concat([c,c.iloc[:1]]))
        with self.assertRaises(AssertionError):cruzar(self.p,c.iloc[1:])

    def test_y_real_distinto_rechazado(self):
        p=self.p.copy();p.loc[p.index[0],'y_real']+=1
        with self.assertRaises(AssertionError):cruzar(p,clasificar(self.s))

    def test_calculo_sin_entrenamiento_y_sin_mutacion(self):
        from contextlib import ExitStack
        copia=self.p.copy(deep=True)
        rutas=['sklearn.linear_model.LinearRegression.fit','sklearn.ensemble.RandomForestRegressor.fit',
               'sklearn.model_selection.GridSearchCV.fit','xgboost.XGBRegressor.fit','statsmodels.tsa.arima.model.ARIMA.fit']
        with ExitStack() as stack:
            for ruta in rutas:stack.enter_context(patch(ruta,side_effect=AssertionError('Reentrenamiento prohibido')))
            tablas,unido,_=calcular(self.s,self.p,self.cov)
        pd.testing.assert_frame_equal(copia,self.p)
        self.assertEqual(tablas['metricas_por_serie.csv'].shape[0],129*4)
        negativas=tablas['auditoria_predicciones_negativas.csv'].set_index('Modelo')
        self.assertEqual(negativas.loc['Regresión Lineal','Predicciones_negativas'],684)

    def test_evidencia_generada(self):
        from scripts.validar_analisis_intermitencia import validar
        validar()

if __name__=='__main__':unittest.main()
