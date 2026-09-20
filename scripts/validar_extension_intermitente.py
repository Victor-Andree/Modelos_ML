"""Verifica la extensión desde evidencia congelada; sin fit ni nuevos pronósticos TEST."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.extension_intermitente import KEY,CLAVES,CORTE,NUMERICAS,EXTRAS,features_intermitentes,escalas_train,metricas,umbral_train,sha

def leer(s,n,fechas=None):return pd.read_csv(s/n,parse_dates=fechas,float_precision='round_trip')
def validar():
    s=ROOT/'resultados/extension_intermitente';b=ROOT/'resultados/semanal'
    config=json.loads((s/'seleccion_train.json').read_text(encoding='utf-8'))
    ejec=json.loads((s/'ejecucion_test.json').read_text(encoding='utf-8'))
    manifest=json.loads((s/'manifiesto_extension.json').read_text(encoding='utf-8'))
    original=json.loads((ROOT/'docs/integridad_previa_extension.json').read_text(encoding='utf-8'))
    for f,h in original['archivos'].items():assert sha(ROOT/f)==h,f
    for f,h in {**manifest['codigo_sha256'],**manifest['salidas_sha256']}.items():assert sha(ROOT/f)==h,f
    assert sha(ROOT/'docs/EXTENSION_DEMANDA_INTERMITENTE.md')==manifest['informe_sha256']
    assert ejec['seleccion_sha256_antes_test']==ejec['seleccion_sha256_despues_test']==sha(s/'seleccion_train.json')
    assert ejec['evaluaciones_predictivas_test']==1 and not ejec['configuracion_revisada_despues_test']
    assert pd.Timestamp(config['seleccion_guardada_utc']) < pd.Timestamp(ejec['test_evaluado_utc'])
    assert pd.Timestamp(config['train_fin'])<CORTE
    train=leer(b,'dataset_train_ml.csv',['semana']);test=leer(b,'dataset_test_ml.csv',['semana'])
    assert len(train)==15170 and len(test)==3940
    weekly=leer(b,'dataset_semanal.csv',['semana'])
    wk_train=weekly.loc[(weekly.semana+pd.Timedelta(days=1)).le(CORTE)]
    escalas=leer(s,'escalas_train.csv')
    pd.testing.assert_frame_equal(escalas_train(wk_train),escalas,check_dtype=False,rtol=1e-12,atol=1e-12)
    # Independent denominator check: no use of test values.
    for _,esc in escalas.iterrows():
        y=wk_train.loc[wk_train.sucursal.eq(esc.sucursal)&wk_train.producto.eq(esc.producto)].sort_values('semana').cantidad.to_numpy()
        delta=np.diff(y)
        if len(delta):np.testing.assert_allclose([esc.escala_abs,esc.escala_cuadrada],[np.mean(abs(delta)),np.mean(delta**2)],rtol=1e-12)
        else:assert np.isnan(esc.escala_abs) and np.isnan(esc.escala_cuadrada)
    f=features_intermitentes(wk_train)
    cruce=train.merge(f,on=KEY,suffixes=('_base','_nuevo'),validate='one_to_one')
    assert len(cruce)==15170
    for col in NUMERICAS:np.testing.assert_allclose(cruce[col+'_base'],cruce[col+'_nuevo'],rtol=0,atol=0)
    oof=leer(s,'predicciones_oof_train.csv',['semana'])
    assert (oof.semana+pd.Timedelta(days=1)).le(CORTE).all()
    assert not oof.duplicated(['Modelo']+KEY).any()
    curvas=leer(s,'cv_umbrales.csv')
    componentes=leer(s,'cv_componentes.csv')
    for columna in ['train_fin','validacion_inicio','validacion_fin']:componentes[columna]=pd.to_datetime(componentes[columna])
    assert (componentes.train_fin<componentes.validacion_inicio).all() and componentes.validacion_fin.lt(CORTE).all()
    for candidato in config['ablacion_features']:
        d=oof.loc[oof.Modelo.eq('TwoStage_'+candidato['features'])]
        mejor,_=umbral_train(d)
        assert mejor['umbral']==candidato['umbral']
        np.testing.assert_allclose(mejor['MSE'],candidato['MSE_cv'],rtol=1e-12)
        for etapa,col in [('clasificacion','clasificador'),('magnitud','regresor')]:
            datos=componentes.loc[componentes.Features.eq(candidato['features'])&componentes.Etapa.eq(etapa)]
            scores=datos.assign(ponderado=datos.Valor*datos.n_validacion).groupby('Candidato')[['ponderado','n_validacion']].sum()
            scores['score']=scores.ponderado/scores.n_validacion
            assert scores.loc[candidato[col],'score'] <= scores.score.min()+1e-12
    assert config['two_stage']==min(config['ablacion_features'],key=lambda x:(x['MSE_cv'],x['features']))
    parametros=leer(s,'cv_baselines_parametros.csv')
    for mod,params in config['baselines'].items():
        d=parametros.loc[parametros.Modelo.eq(mod)]
        assert json.loads(d.loc[d.MSE_cv.idxmin(),'parametros'])==params
    nuevas=leer(s,'predicciones_nuevas.csv',['semana'])
    comunes=leer(s,'predicciones_comparacion_comun.csv',['semana'])
    base=leer(b,'predicciones_comunes.csv',['semana'])
    tabla=leer(s,'comparacion_extension.csv').set_index('Modelo')
    ref=base.loc[base.Modelo.eq('Regresión Lineal'),KEY+['y_real']].sort_values(KEY).reset_index(drop=True)
    assert len(tabla)==10 and tabla.index.is_unique and comunes.Modelo.nunique()==10
    for mod,d in nuevas.groupby('Modelo'):
        assert len(d)==3940 and not d.duplicated(KEY).any()
        original_test=test[KEY+['cantidad']].rename(columns={'cantidad':'y_real'}).sort_values(KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(d[KEY+['y_real']].sort_values(KEY).reset_index(drop=True),original_test,check_dtype=False,check_exact=True)
    for mod,d in comunes.groupby('Modelo'):
        d=d.sort_values(KEY).reset_index(drop=True)
        assert len(d)==3444 and len(d[CLAVES].drop_duplicates())==129
        pd.testing.assert_frame_equal(d[KEY+['y_real']],ref,check_dtype=False,check_exact=True)
        if mod in set(base.Modelo):
            orig=base.loc[base.Modelo.eq(mod)].sort_values(KEY)
            np.testing.assert_array_equal(orig.y_pred,d.y_pred)
        else:
            orig=d[KEY].merge(nuevas.loc[nuevas.Modelo.eq(mod)],on=KEY,validate='one_to_one')
            np.testing.assert_array_equal(orig.y_pred,d.y_pred)
        m=metricas(d,escalas)
        for col in ['MAE','RMSE','R2','MASE','RMSSE']:np.testing.assert_allclose(m[col],tabla.loc[mod,col],rtol=1e-12,atol=1e-12)
        # Fully independent macro scaled errors from per-series test residuals.
        mases=[];rmsses=[]
        for (suc,prod),parte in d.groupby(CLAVES):
            tr=wk_train.loc[wk_train.sucursal.eq(suc)&wk_train.producto.eq(prod)].sort_values('semana').cantidad.to_numpy()
            delta=np.diff(tr);e=parte.y_real.to_numpy()-parte.y_pred.to_numpy()
            if len(delta) and np.mean(abs(delta))>0:
                mases.append(np.mean(abs(e))/np.mean(abs(delta)))
                rmsses.append(np.sqrt(np.mean(e**2)/np.mean(delta**2)))
        np.testing.assert_allclose([np.mean(mases),np.mean(rmsses)],[tabla.loc[mod,'MASE'],tabla.loc[mod,'RMSSE']],rtol=1e-12)
    dos=leer(s,'two_stage_resultados.csv',['semana'])
    assert dos.umbral.eq(config['two_stage']['umbral']).all()
    np.testing.assert_array_equal(dos.y_pred,np.where(dos.probabilidad>=config['two_stage']['umbral'],dos.magnitud,0))
    assert dos.magnitud.ge(0).all()
    # CV fold summary and scaled errors can also be recalculated from OOF and fold-specific history.
    cv=leer(s,'metricas_cv.csv')
    for _,row in cv.iterrows():
        d=oof.loc[oof.Modelo.eq(row.Modelo)&oof.fold.eq(row.fold)]
        sc=escalas_train(wk_train.loc[wk_train.semana.le(pd.Timestamp(row.train_fin))])
        m=metricas(d,sc)
        for col in ['MAE','RMSE','R2','MASE','RMSSE']:np.testing.assert_allclose(m[col],row[col],rtol=1e-12,atol=1e-12)
    assert leer(s,'auditoria_leakage.csv').Resultado.eq('OK').all()
    print(f'OK extensión: configuración TRAIN congelada, TEST único, 10 modelos x 3444 claves, escalas sólo TRAIN, métricas reproducibles y {len(original["archivos"])} archivos previos intactos.')

if __name__=='__main__':validar()
