"""Verifica tablas publicadas mediante recálculo desde las predicciones inmutables."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analisis_intermitencia import calcular, huella, MODELOS, validar_predicciones

def validar():
    salida=ROOT/'resultados/semanal'
    p=pd.read_csv(salida/'predicciones_comunes.csv',parse_dates=['semana'],float_precision='round_trip')
    s=pd.read_csv(salida/'dataset_semanal.csv',parse_dates=['semana'],float_precision='round_trip')
    cov=pd.read_csv(salida/'resumen_cobertura.csv',float_precision='round_trip')
    validar_predicciones(p)
    esperadas,unido,bordes=calcular(s,p,cov)
    for nombre,esperado in esperadas.items():
        actual=pd.read_csv(salida/nombre,float_precision='round_trip',keep_default_na=True)
        for col in esperado.columns:
            if pd.api.types.is_datetime64_any_dtype(esperado[col]):
                actual[col]=pd.to_datetime(actual[col])
            elif pd.api.types.is_string_dtype(esperado[col]) and esperado[col].fillna('').eq('').any():
                actual[col]=actual[col].fillna('')
                esperado[col]=esperado[col].fillna('')
        pd.testing.assert_frame_equal(esperado.reset_index(drop=True),actual,check_dtype=False,rtol=1e-12,atol=1e-12)
    # Independent metric formulas against every subgroup, not only the shared helper.
    mappings={'metricas_por_intermitencia.csv':('Clasificacion_demanda','clasificacion_demanda'),
              'metricas_por_grupo_demanda.csv':('grupo_demanda','grupo_demanda'),
              'metricas_por_ceros.csv':('intervalo_ceros','intervalo_ceros'),
              'metricas_cero_vs_positivo.csv':('Tipo_demanda','Tipo_demanda')}
    for nombre,(etiqueta,columna) in mappings.items():
        for _,row in pd.read_csv(salida/nombre).iterrows():
            d=unido.loc[unido.Modelo.eq(row.Modelo)&unido[columna].eq(row[etiqueta])]
            assert len(d)==row.Observaciones
            assert len(d[['sucursal','producto']].drop_duplicates())==row.Series
            if not len(d):
                assert np.isnan(row.MAE) and np.isnan(row.RMSE) and np.isnan(row.R2)
                continue
            errores=(d.y_real-d.y_pred).to_numpy()
            np.testing.assert_allclose([np.mean(abs(errores)),np.linalg.norm(errores)/np.sqrt(len(d))],[row.MAE,row.RMSE],rtol=1e-12,atol=1e-12)
            sst=sum((float(y)-float(d.y_real.mean()))**2 for y in d.y_real)
            if len(d)<2 or sst==0: assert np.isnan(row.R2)
            else: np.testing.assert_allclose(1-sum(float(e)**2 for e in errores)/sst,row.R2,rtol=1e-12,atol=1e-12)
    contrib=pd.read_csv(salida/'contribuciones_error.csv').groupby('Modelo')[['Contribucion_MAE','Contribucion_MSE']].sum()
    globales=pd.read_csv(salida/'comparacion_modelos_comun.csv').set_index('Modelo')
    for mod in MODELOS:
        np.testing.assert_allclose(contrib.loc[mod,'Contribucion_MAE'],globales.loc[mod,'MAE'],rtol=1e-12)
        np.testing.assert_allclose(contrib.loc[mod,'Contribucion_MSE'],globales.loc[mod,'RMSE']**2,rtol=1e-12)
    baseline=json.loads((ROOT/'docs/integridad_previa_intermitencia.json').read_text(encoding='utf-8'))
    for ruta,sha in baseline['archivos'].items():assert huella(ROOT/ruta)==sha,ruta
    m=json.loads((salida/'experimento_intermitencia.json').read_text(encoding='utf-8'))
    assert m['observaciones_comunes']==3444 and m['series_comunes']==129
    assert not m['entrenamiento_realizado'] and not m['predicciones_modificadas'] and not m['app_modificada']
    assert m['intervalos_ceros']==bordes
    for ruta,sha in {**m['entradas_sha256'],**m['codigo_sha256'],**m['salidas_sha256']}.items():assert huella(ROOT/ruta)==sha,ruta
    print(f'OK: 3444 claves x 4 modelos, 129 series, clasificación única, métricas reproducibles, {len(baseline["archivos"])} archivos originales intactos, figuras e informe verificados.')

if __name__=='__main__':validar()
