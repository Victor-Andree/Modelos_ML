"""Extensión aislada: selección exclusivamente temporal TRAIN, evaluación bloqueada."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import itertools
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from src.preprocessing.demanda_semanal import agregar_features, particiones_por_semana, CLAVES, NUMERICAS

KEY=CLAVES+['semana']
CORTE=pd.Timestamp('2026-01-01')
EXTRAS=['semanas_desde_ultima_demanda','numero_eventos_positivos_ultimas_4',
        'numero_eventos_positivos_ultimas_8','tasa_demanda_positiva_ultimas_4',
        'tasa_demanda_positiva_ultimas_8','media_demanda_positiva_historica',
        'demanda_no_cero_lag_1','demanda_no_cero_lag_2']
ALPHAS=[0.05,0.1,0.2]
UMBRALES=[0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
NOMBRES=['Logistic Regression','Random Forest','XGBoost']
REGRESORES=['Regresión Lineal','Random Forest','XGBoost']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guardar(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def features_intermitentes(semanal):
    f=agregar_features(semanal)
    partes=[]
    for _,d in f.groupby(CLAVES,observed=True):
        d=d.sort_values('semana').copy()
        y=d.cantidad.to_numpy()
        ultima=None;eventos=[];suma=0.;dist=[];medias=[];n1=[];n2=[]
        for i,v in enumerate(y):
            dist.append(i-ultima if ultima is not None else i+1)
            medias.append(suma/len(eventos) if eventos else 0.)
            n1.append(eventos[-1] if eventos else 0.)
            n2.append(eventos[-2] if len(eventos)>1 else 0.)
            if v>0:ultima=i;eventos.append(float(v));suma+=float(v)
        d['semanas_desde_ultima_demanda']=dist
        d['media_demanda_positiva_historica']=medias
        d['demanda_no_cero_lag_1']=n1;d['demanda_no_cero_lag_2']=n2
        for w in [4,8]:
            eventos_previos=d.cantidad.gt(0).astype(float).shift(1).rolling(w,min_periods=w).sum()
            d[f'numero_eventos_positivos_ultimas_{w}']=eventos_previos
            d[f'tasa_demanda_positiva_ultimas_{w}']=eventos_previos/w
        partes.append(d)
    return pd.concat(partes,ignore_index=True).sort_values(KEY).reset_index(drop=True)

def predecir_serie(y,metodo,alpha=.1,beta=.1):
    """Emitir pronóstico antes de actualizar con y_t, desde arranque cero."""
    assert metodo in ['Cero','Naive_ultimo','Croston','SBA','TSB']
    z=0.;intervalo=1.;prob=0.;desde=0;inicializado=False;ultimo=0.;pred=[]
    for i,valor in enumerate(np.asarray(y,dtype=float)):
        if metodo=='Cero':f=0.
        elif metodo=='Naive_ultimo':f=ultimo
        elif metodo=='TSB':f=z*prob if inicializado else 0.
        else:f=z/intervalo*(1-alpha/2 if metodo=='SBA' else 1) if inicializado else 0.
        pred.append(f)
        desde+=1
        if not inicializado:
            if valor>0:
                z=valor;intervalo=float(desde);prob=1/(i+1);desde=0;inicializado=True
        else:
            prob=(1-beta)*prob+beta*float(valor>0)
            if valor>0:
                z=(1-alpha)*z+alpha*valor
                intervalo=(1-alpha)*intervalo+alpha*desde
                desde=0
        ultimo=valor
    return np.asarray(pred)

def baseline(semanal,metodo,params):
    partes=[]
    for _,d in semanal.groupby(CLAVES,observed=True):
        d=d.sort_values('semana').copy()
        d['y_pred']=predecir_serie(d.cantidad,metodo,**params)
        partes.append(d[KEY+['cantidad','y_pred']].rename(columns={'cantidad':'y_real'}))
    return pd.concat(partes,ignore_index=True)

def escalas_train(semanal_train):
    assert (semanal_train.semana+pd.Timedelta(days=1)).le(CORTE).all()
    filas=[]
    for claves,d in semanal_train.groupby(CLAVES,observed=True):
        delta=d.sort_values('semana').cantidad.diff().dropna().to_numpy()
        filas.append(dict(zip(CLAVES,claves)) | {'escala_abs':float(np.abs(delta).mean()) if len(delta) else np.nan,
            'escala_cuadrada':float(np.square(delta).mean()) if len(delta) else np.nan,'semanas_train_escala':len(d)})
    return pd.DataFrame(filas)

def metricas(d,escalas):
    if not len(d):return {'MAE':np.nan,'RMSE':np.nan,'R2':np.nan,'MASE':np.nan,'RMSSE':np.nan,
        'observaciones':0,'series':0,'series_escala_valida':0,'observaciones_escala_valida':0,'R2_estado':'sin_datos'}
    e=d.y_real.to_numpy()-d.y_pred.to_numpy();sst=np.square(d.y_real-d.y_real.mean()).sum()
    por=d.assign(ae=np.abs(e),se=np.square(e)).groupby(CLAVES,observed=True).agg(
        MAE_serie=('ae','mean'),MSE_serie=('se','mean'),N=('ae','size')).reset_index().merge(escalas,on=CLAVES,how='left',validate='one_to_one')
    valida=por.escala_abs.gt(0)&por.escala_cuadrada.gt(0)
    v=por.loc[valida]
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.square(e).mean())),
        'R2':float(1-np.square(e).sum()/sst) if len(d)>1 and sst>0 else np.nan,
        'MASE':float((v.MAE_serie/v.escala_abs).mean()) if len(v) else np.nan,
        'RMSSE':float(np.sqrt(v.MSE_serie/v.escala_cuadrada).mean()) if len(v) else np.nan,
        'observaciones':len(d),'series':len(por),'series_escala_valida':len(v),
        'observaciones_escala_valida':int(v.N.sum()),'R2_estado':'definido' if len(d)>1 and sst>0 else 'no_definido'}

def estimador(nombre,tipo,numericas):
    if tipo=='clasificacion':
        opciones={'Logistic Regression':LogisticRegression(C=1.,max_iter=2000,random_state=42),
            'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=2,random_state=42,n_jobs=-1),
            'XGBoost':XGBClassifier(n_estimators=100,max_depth=3,learning_rate=.05,objective='binary:logistic',eval_metric='logloss',random_state=42,n_jobs=-1)}
    else:
        opciones={'Regresión Lineal':LinearRegression(),
            'Random Forest':RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=2,random_state=42,n_jobs=-1),
            'XGBoost':XGBRegressor(n_estimators=100,max_depth=3,learning_rate=.05,objective='reg:squarederror',random_state=42,n_jobs=-1)}
    pre=ColumnTransformer([('categorias',OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False),CLAVES),
        ('numericas',StandardScaler() if nombre=='Logistic Regression' else 'passthrough',numericas)])
    return Pipeline([('preprocesador',pre),('estimador',opciones[nombre])])

def umbral_train(oof):
    assert oof.semana.lt(CORTE).all(), 'El umbral sólo acepta validaciones TRAIN'
    filas=[]
    for u in UMBRALES:
        p=np.where(oof.probabilidad>=u,oof.magnitud,0.)
        filas.append({'umbral':u,'MSE':float(np.mean(np.square(oof.y_real-p))),
                      'MAE':float(np.mean(np.abs(oof.y_real-p)))})
    mejor=min(filas,key=lambda x:(x['MSE'],x['umbral']))
    return mejor,pd.DataFrame(filas)

def seleccionar(train,semanal_train):
    """No recibe archivos, TEST ni métricas del holdout."""
    assert len(train)==15170
    assert (train.semana+pd.Timedelta(days=1)).le(CORTE).all()
    assert (semanal_train.semana+pd.Timedelta(days=1)).le(CORTE).all()
    folds=particiones_por_semana(train.semana,n_splits=3)
    registros=[];componentes=[];candidatos=[];oof_features={};curvas=[]
    for tipo_features,numericas in [('basicas',NUMERICAS),('intermitentes',NUMERICAS+EXTRAS)]:
        partes={('clasificacion',n):[] for n in NOMBRES}|{('magnitud',n):[] for n in REGRESORES}
        for fold,(a,b) in enumerate(folds,1):
            tr=train.iloc[a];va=train.iloc[b]
            assert tr.semana.max()<va.semana.min()
            for tipo,nombres in [('clasificacion',NOMBRES),('magnitud',REGRESORES)]:
                for nombre in nombres:
                    modelo=estimador(nombre,tipo,numericas)
                    entrenamiento=tr if tipo=='clasificacion' else tr.loc[tr.cantidad.gt(0)]
                    target=entrenamiento.cantidad.gt(0).astype(int) if tipo=='clasificacion' else entrenamiento.cantidad
                    modelo.fit(entrenamiento[CLAVES+numericas],target)
                    pred=modelo.predict_proba(va[CLAVES+numericas])[:,1] if tipo=='clasificacion' else np.maximum(0,modelo.predict(va[CLAVES+numericas]))
                    detalle=va[KEY].copy();detalle['y_real']=va.cantidad.to_numpy();detalle['valor']=pred;detalle['fold']=fold
                    partes[(tipo,nombre)].append(detalle)
                    mask=np.ones(len(va),dtype=bool) if tipo=='clasificacion' else va.cantidad.gt(0).to_numpy()
                    y=va.cantidad.gt(0).astype(float).to_numpy() if tipo=='clasificacion' else va.cantidad.to_numpy()
                    score=float(np.square(y[mask]-pred[mask]).mean())
                    componentes.append({'Features':tipo_features,'Etapa':tipo,'Candidato':nombre,'fold':fold,
                        'Metrica':'Brier' if tipo=='clasificacion' else 'MSE_positivas','Valor':score,
                        'n_validacion':int(mask.sum()),'train_fin':str(tr.semana.max()),'validacion_inicio':str(va.semana.min()),'validacion_fin':str(va.semana.max())})
            print('CV',tipo_features,'fold',fold,'terminado',flush=True)
        juntos={k:pd.concat(v,ignore_index=True) for k,v in partes.items()}
        def score(k):
            d=juntos[k];y=d.y_real.gt(0).astype(float) if k[0]=='clasificacion' else d.y_real
            mask=np.ones(len(d),dtype=bool) if k[0]=='clasificacion' else d.y_real.gt(0)
            return float(np.square(y[mask]-d.loc[mask,'valor']).mean())
        clf=min(NOMBRES,key=lambda n:score(('clasificacion',n)))
        reg=min(REGRESORES,key=lambda n:score(('magnitud',n)))
        d=juntos[('clasificacion',clf)].rename(columns={'valor':'probabilidad'})
        m=juntos[('magnitud',reg)]
        pd.testing.assert_frame_equal(d[KEY+['y_real','fold']],m[KEY+['y_real','fold']])
        d['magnitud']=m.valor.to_numpy()
        mejor,curva=umbral_train(d);curva['Features']=tipo_features;curvas.append(curva)
        d['y_pred']=np.where(d.probabilidad>=mejor['umbral'],d.magnitud,0.)
        d['Modelo']='TwoStage_'+tipo_features
        oof_features[tipo_features]=d
        candidatos.append({'features':tipo_features,'clasificador':clf,'regresor':reg,
            'umbral':mejor['umbral'],'MSE_cv':mejor['MSE'],'MAE_cv':mejor['MAE']})
    elegido=min(candidatos,key=lambda x:(x['MSE_cv'],x['features']))
    # Each method uses a single globally selected parameter set, not per-series TEST tuning.
    baselines={};oof_baselines={};param_scores=[]
    for metodo in ['Cero','Naive_ultimo','Croston','SBA','TSB']:
        rejilla=[{}] if metodo in ['Cero','Naive_ultimo'] else ([{'alpha':a,'beta':b} for a,b in itertools.product(ALPHAS,ALPHAS)] if metodo=='TSB' else [{'alpha':a} for a in ALPHAS])
        opciones=[]
        for params in rejilla:
            pred=baseline(semanal_train,metodo,params)
            lista=[]
            for fold,(a,b) in enumerate(folds,1):
                d=train.iloc[b][KEY].merge(pred,on=KEY,validate='one_to_one');d['fold']=fold;d['Modelo']=metodo;lista.append(d)
            d=pd.concat(lista,ignore_index=True)
            score=float(np.square(d.y_real-d.y_pred).mean())
            opciones.append((score,params,d));param_scores.append({'Modelo':metodo,'parametros':json.dumps(params,sort_keys=True),'MSE_cv':score})
        mejor=min(opciones,key=lambda x:x[0]);baselines[metodo]=mejor[1];oof_baselines[metodo]=mejor[2]
    for metodo,d in {**oof_baselines,**{'TwoStage_'+n:d for n,d in oof_features.items()}}.items():
        for fold,(a,b) in enumerate(folds,1):
            fin=train.iloc[a].semana.max()
            esc=escalas_train(semanal_train.loc[semanal_train.semana.le(fin)])
            parte=d.loc[d.fold.eq(fold)]
            registros.append({'Modelo':metodo,'fold':fold,**metricas(parte,esc),
                'train_fin':str(fin),'validacion_inicio':str(train.iloc[b].semana.min()),'validacion_fin':str(train.iloc[b].semana.max())})
    config={'two_stage':elegido,'ablacion_features':candidatos,'baselines':baselines,
        'regla_seleccion':'clasificador por Brier OOF; magnitud por MSE OOF en positivas; umbral y features por MSE final OOF',
        'origen_seleccion':'TRAIN/CV temporal exclusivamente','fecha_corte':str(CORTE.date()),'n_splits':3,
        'train_filas':len(train),'train_fin':str(train.semana.max()),'umbrales_candidatos':UMBRALES,
        'alphas_candidatos':ALPHAS,'semilla':42,'seleccion_guardada_utc':datetime.now(timezone.utc).isoformat(),
        'regla_magnitud':'max(0, predicción de magnitud) aplicada igual en CV y TEST, fijada antes de evaluar'}
    oof=pd.concat(list(oof_baselines.values())+list(oof_features.values()),ignore_index=True)
    return config,pd.DataFrame(registros),pd.DataFrame(componentes),pd.concat(curvas),pd.DataFrame(param_scores),oof

def auditar_features(semanal):
    original=features_intermitentes(semanal)
    limites=[pd.Timestamp('2025-06-01'),pd.Timestamp('2026-01-11')]
    filas=[]
    for limite in limites:
        mutado=semanal.copy();mutado.loc[mutado.semana.ge(limite),'cantidad']=99999.
        despues=features_intermitentes(mutado)
        antes=original.loc[original.semana.le(limite)]
        futuro=despues.loc[despues.semana.le(limite)]
        for col in NUMERICAS+EXTRAS:
            np.testing.assert_allclose(antes[col],futuro[col],equal_nan=True,rtol=0,atol=0)
            filas.append({'Comprobacion':'perturbar_target_actual_y_futuro','Feature':col,'Fecha':str(limite.date()),'Resultado':'OK'})
    return pd.DataFrame(filas)

def ejecutar(root):
    root=Path(root);sal=root/'resultados/extension_intermitente';base=root/'resultados/semanal'
    sal.mkdir(parents=True,exist_ok=True)
    if (sal/'predicciones_nuevas.csv').exists():
        raise RuntimeError('TEST ya evaluado: no se repite tuning ni predicción. Recalcular sólo reportes desde predicciones existentes.')
    proteger=json.loads((root/'docs/integridad_previa_extension.json').read_text(encoding='utf-8'))
    for p,h in proteger['archivos'].items():assert sha(root/p)==h,p
    semanal=pd.read_csv(base/'dataset_semanal.csv',parse_dates=['semana'],float_precision='round_trip')
    semanal_train=semanal.loc[(semanal.semana+pd.Timedelta(days=1)).le(CORTE)].copy()
    train_keys=pd.read_csv(base/'dataset_train_ml.csv',parse_dates=['semana'],float_precision='round_trip')
    train=train_keys[KEY].merge(features_intermitentes(semanal_train),on=KEY,validate='one_to_one').sort_values(['semana']+CLAVES).reset_index(drop=True)
    assert len(train)==15170 and not train[NUMERICAS+EXTRAS].isna().any().any()
    config,cv,componentes,umbrales,parametros,oof=seleccionar(train,semanal_train)
    config['sha256_train']=sha(base/'dataset_train_ml.csv')
    guardar(sal/'seleccion_train.json',config)
    sello=sha(sal/'seleccion_train.json')
    for nombre,d in [('metricas_cv.csv',cv),('cv_componentes.csv',componentes),('cv_umbrales.csv',umbrales),
                     ('cv_baselines_parametros.csv',parametros),('predicciones_oof_train.csv',oof)]:d.to_csv(sal/nombre,index=False)
    cv.groupby('Modelo')[['MAE','RMSE','R2','MASE','RMSSE']].agg(['mean','std']).to_csv(sal/'resumen_cv.csv')
    f=config['two_stage'];nums=NUMERICAS+(EXTRAS if f['features']=='intermitentes' else [])
    clf=estimador(f['clasificador'],'clasificacion',nums);reg=estimador(f['regresor'],'magnitud',nums)
    clf.fit(train[CLAVES+nums],train.cantidad.gt(0).astype(int))
    positivas=train.loc[train.cantidad.gt(0)]
    reg.fit(positivas[CLAVES+nums],positivas.cantidad)
    joblib.dump({'clasificador':clf,'magnitud':reg,'features':CLAVES+nums,'umbral':f['umbral']},sal/'modelo_two_stage.pkl')
    # Test rows are only read after saving the immutable selection.
    test_original=pd.read_csv(base/'dataset_test_ml.csv',parse_dates=['semana'],float_precision='round_trip')
    test=test_original[KEY].merge(features_intermitentes(semanal),on=KEY,validate='one_to_one').sort_values(['semana']+CLAVES).reset_index(drop=True)
    assert len(test)==3940 and (test.semana-pd.Timedelta(days=6)).ge(CORTE).all()
    pd.testing.assert_frame_equal(test[KEY+['cantidad']].reset_index(drop=True),test_original.sort_values(['semana']+CLAVES)[KEY+['cantidad']].reset_index(drop=True),check_dtype=False)
    p=clf.predict_proba(test[CLAVES+nums])[:,1];m=np.maximum(0,reg.predict(test[CLAVES+nums]))
    dos=test[KEY].copy();dos['y_real']=test.cantidad;dos['probabilidad']=p;dos['magnitud']=m;dos['umbral']=f['umbral'];dos['y_pred']=np.where(p>=f['umbral'],m,0.);dos['Modelo']='TwoStage'
    dos.to_csv(sal/'two_stage_resultados.csv',index=False)
    nuevas=[dos[KEY+['y_real','y_pred','Modelo']]]
    for metodo,params in config['baselines'].items():
        d=test[KEY].merge(baseline(semanal,metodo,params),on=KEY,validate='one_to_one');d['Modelo']=metodo;nuevas.append(d)
    pred=pd.concat(nuevas,ignore_index=True)
    pred['error']=pred.y_real-pred.y_pred;pred['error_absoluto']=pred.error.abs()
    pred.to_csv(sal/'predicciones_nuevas.csv',index=False)
    pred.loc[pred.Modelo.ne('TwoStage')].to_csv(sal/'predicciones_baselines.csv',index=False)
    escalas_train(semanal_train).to_csv(sal/'escalas_train.csv',index=False)
    auditar_features(semanal).to_csv(sal/'auditoria_leakage.csv',index=False)
    assert sello==sha(sal/'seleccion_train.json')
    guardar(sal/'ejecucion_test.json',{'seleccion_sha256_antes_test':sello,'seleccion_sha256_despues_test':sha(sal/'seleccion_train.json'),
        'evaluaciones_predictivas_test':1,'test_evaluado_utc':datetime.now(timezone.utc).isoformat(),
        'configuracion_revisada_despues_test':False,'predicciones_nuevas_sha256':sha(sal/'predicciones_nuevas.csv')})
    from src.informe_extension import resumir_extension
    resumir_extension(root)
    for p,h in proteger['archivos'].items():assert sha(root/p)==h,p
    print('Extensión terminada: baseline intacto y configuración seleccionada sólo en TRAIN.',flush=True)
