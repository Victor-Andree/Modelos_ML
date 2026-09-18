"""Análisis descriptivo; sólo consume predicciones ya calculadas. No entrena modelos."""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd

KEY = ['sucursal','producto','semana']
SERIE = KEY[:2]
MODELOS = ['Regresión Lineal','Random Forest','XGBoost','ARIMA (Optimizado ADF/AIC)']
PATRONES = ['suave','erratica','intermitente','lumpy','positivas_insuficientes','sin_demanda_positiva','sin_historia_train']
GRUPOS = ['menor_intermitencia','mayor_intermitencia','sin_demanda_positiva','sin_historia_train']
CORTE = pd.Timestamp('2026-01-01')
ADI_CORTE, CV2_CORTE = 1.32, 0.49
FUENTE = 'https://doi.org/10.1057/palgrave.jors.2601841'


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def clasificar(semanal):
    """ADI=N/N+; CV2=varianza muestral positiva / media positiva², sólo TRAIN."""
    semanal = semanal.copy()
    semanal['semana'] = pd.to_datetime(semanal.semana)
    assert not semanal.duplicated(KEY).any()
    assert semanal.cantidad.notna().all() and semanal.cantidad.ge(0).all()
    filas = []
    for (sucursal,producto), d in semanal.groupby(SERIE, observed=True):
        # The whole weekly interval must precede cutoff; no cross-cutoff week.
        train = d.loc[(d.semana + pd.Timedelta(days=1)).le(CORTE)].sort_values('semana')
        y = train.cantidad
        positiva = y[y.gt(0)]
        n, npos = len(y), len(positiva)
        media = float(positiva.mean()) if npos else np.nan
        desviacion = float(positiva.std(ddof=1)) if npos >= 2 else np.nan
        adi = n / npos if npos else np.nan
        cv2 = (desviacion / media)**2 if npos >= 2 else np.nan
        if not n:
            patron = grupo = 'sin_historia_train'
        elif not npos:
            patron = grupo = 'sin_demanda_positiva'
        else:
            grupo = 'menor_intermitencia' if adi < ADI_CORTE else 'mayor_intermitencia'
            if npos < 2:
                patron = 'positivas_insuficientes'
            elif adi < ADI_CORTE:
                patron = 'suave' if cv2 < CV2_CORTE else 'erratica'
            else:
                patron = 'intermitente' if cv2 < CV2_CORTE else 'lumpy'
        filas.append({'sucursal':sucursal,'producto':producto,'ADI':adi,'CV2':cv2,
            'porcentaje_ceros':100 * y.eq(0).mean() if n else np.nan,
            'clasificacion_demanda':patron,'grupo_demanda':grupo,'n_semanas':n,
            'n_semanas_demanda_positiva':npos,'media_demanda_positiva':media,
            'desviacion_demanda_positiva':desviacion,'periodo_clasificacion':'TRAIN anterior a 2026-01-01',
            'inicio_train':train.semana.min(),'fin_train':train.semana.max(),
            'CV2_estimable':npos >= 2})
    return pd.DataFrame(filas)


def validar_predicciones(p, n_esperado=3444, series_esperadas=129):
    assert set(p.Modelo) == set(MODELOS), 'Falta un modelo'
    assert p[KEY + ['Modelo','y_real','y_pred']].notna().all().all()
    assert np.isfinite(p[['y_real','y_pred']]).all().all()
    assert p.y_real.ge(0).all()
    assert not p.duplicated(['Modelo'] + KEY).any()
    referencia = None
    for modelo in MODELOS:
        d = p.loc[p.Modelo.eq(modelo)].sort_values(KEY).reset_index(drop=True)
        assert len(d) == n_esperado
        assert len(d[SERIE].drop_duplicates()) == series_esperadas
        comparables = d[KEY + ['y_real']]
        if referencia is None: referencia = comparables
        else: pd.testing.assert_frame_equal(referencia, comparables, check_dtype=False, check_exact=True)
    return referencia


def cruzar(p, clasificacion, n_esperado=3444, series_esperadas=129):
    validar_predicciones(p,n_esperado,series_esperadas)
    assert not clasificacion.duplicated(SERIE).any()
    unido = p.merge(clasificacion,on=SERIE,how='left',validate='many_to_one',indicator=True)
    assert len(unido) == len(p) and unido._merge.eq('both').all()
    assert unido.clasificacion_demanda.notna().all() and unido.grupo_demanda.notna().all()
    unido = unido.drop(columns='_merge')
    for categoria in ['clasificacion_demanda','grupo_demanda']:
        for _, d in unido.groupby(categoria,observed=True):
            referencia = None
            for modelo in MODELOS:
                actual = d.loc[d.Modelo.eq(modelo),KEY + ['y_real']].sort_values(KEY).reset_index(drop=True)
                if referencia is None: referencia=actual
                else: pd.testing.assert_frame_equal(referencia,actual,check_dtype=False,check_exact=True)
    return unido


def metricas(d, varianza_global=None, ceros_global=None):
    """R² indefinido para target constante o n<2, sin imputar un 0 o 1."""
    n = len(d)
    if not n:
        return {'MAE':np.nan,'RMSE':np.nan,'R2':np.nan,'Observaciones':0,'Series':0,
                'Varianza_y_real':np.nan,'SST':np.nan,'Porcentaje_ceros_test':np.nan,
                'R2_estado':'sin_observaciones','Advertencias':'grupo_ausente'}
    y = d.y_real.to_numpy(dtype=float)
    error = y - d.y_pred.to_numpy(dtype=float)
    sst = float(np.square(y-y.mean()).sum())
    var = sst/n
    pct = float(100*np.mean(y==0))
    estado = 'no_definido_n_menor_2' if n<2 else ('no_definido_target_constante' if sst==0 else 'definido')
    alertas=[]
    if varianza_global is not None and 0 < var < varianza_global:
        alertas.append('varianza_menor_que_global')
    if ceros_global is not None and pct > ceros_global:
        alertas.append('porcentaje_ceros_mayor_que_global')
    if len(d[SERIE].drop_duplicates())==1: alertas.append('una_sola_serie')
    return {'MAE':float(np.abs(error).mean()),'RMSE':float(np.sqrt(np.square(error).mean())),
        'R2':float(1-np.square(error).sum()/sst) if estado=='definido' else np.nan,
        'Observaciones':n,'Series':len(d[SERIE].drop_duplicates()),'Varianza_y_real':var,'SST':sst,
        'Porcentaje_ceros_test':pct,'R2_estado':estado,'Advertencias':';'.join(alertas)}


def resumir(d, columna, categorias=None, nombre_columna=None):
    referencia=d.loc[d.Modelo.eq(MODELOS[0])]
    var=referencia.y_real.var(ddof=0)
    ceros=100*referencia.y_real.eq(0).mean()
    filas=[]
    for cat in (categorias if categorias is not None else sorted(d[columna].dropna().unique())):
        for modelo in MODELOS:
            parte=d.loc[d[columna].eq(cat) & d.Modelo.eq(modelo)]
            filas.append({'Modelo':modelo, nombre_columna or columna:cat, **metricas(parte,var,ceros)})
    return pd.DataFrame(filas)


def intervalos_ceros(clasificacion_comun):
    """Cuartiles entre series comunes, sin ponderar por número de filas test.

    Cortes repetidos se colapsan; no se usa ningún error para definir intervalos.
    """
    d=clasificacion_comun.copy()
    validas=d.porcentaje_ceros.notna()
    bordes=np.unique(d.loc[validas,'porcentaje_ceros'].quantile([0,.25,.5,.75,1]).to_numpy())
    d['intervalo_ceros']='sin_historia_train'
    if len(bordes)==1:
        d.loc[validas,'intervalo_ceros']='Q1'
        descripcion=[{'intervalo_ceros':'Q1','limite_inferior':float(bordes[0]),'limite_superior':float(bordes[0]),'inferior_inclusivo':True}]
    else:
        etiquetas=[f'Q{i+1}' for i in range(len(bordes)-1)]
        d.loc[validas,'intervalo_ceros']=pd.cut(d.loc[validas,'porcentaje_ceros'],bins=bordes,
            labels=etiquetas,include_lowest=True,right=True).astype(str)
        descripcion=[{'intervalo_ceros':etiquetas[i],'limite_inferior':float(bordes[i]),
            'limite_superior':float(bordes[i+1]),'inferior_inclusivo':i==0} for i in range(len(etiquetas))]
    assert d.intervalo_ceros.notna().all()
    return d,descripcion


def auditoria_negativas(p):
    filas=[]
    for modelo in MODELOS:
        d=p.loc[p.Modelo.eq(modelo)];neg=d.loc[d.y_pred.lt(0)]
        m=metricas(neg)
        filas.append({'Modelo':modelo,'Predicciones_negativas':len(neg),
            'Porcentaje_predicciones_negativas':100*len(neg)/len(d),'Minimo_predicho':d.y_pred.min(),
            'MAE_negativas':m['MAE'],'RMSE_negativas':m['RMSE'],
            'Negativas_con_y_cero':int(neg.y_real.eq(0).sum()),
            'Negativas_con_y_positivo':int(neg.y_real.gt(0).sum()),
            'Mediana_absoluta_negativas':neg.y_pred.abs().median(),
            'Percentil95_absoluto_negativas':neg.y_pred.abs().quantile(.95)})
    return pd.DataFrame(filas)


def conteo_minimos(serie, modelos=MODELOS):
    filas=[]
    for metrica in ['MAE','RMSE']:
        valores=serie.pivot(index=SERIE,columns='Modelo',values=metrica)[modelos]
        minimo=valores.min(axis=1).to_numpy()
        empatados=np.isclose(valores.to_numpy(),minimo[:,None],rtol=1e-9,atol=1e-12)
        for j,modelo in enumerate(modelos):
            filas.append({'Modelo':modelo,'Metrica':metrica,
                'Series_minimo_exclusivo':int((empatados[:,j] & (empatados.sum(axis=1)==1)).sum()),
                'Series_minimo_compartido':int((empatados[:,j] & (empatados.sum(axis=1)>1)).sum()),
                'Series_minimo_incluyendo_empates':int(empatados[:,j].sum()),'Series_totales':len(valores)})
    return pd.DataFrame(filas)


def calcular(semanal, predicciones, cobertura):
    referencia=validar_predicciones(predicciones)
    clasificacion=clasificar(semanal)
    comunes=referencia[SERIE].drop_duplicates()
    c=comunes.merge(clasificacion,on=SERIE,validate='one_to_one')
    assert len(c)==129
    c,bordes=intervalos_ceros(c)
    clasificacion=clasificacion.merge(c[SERIE+['intervalo_ceros']],on=SERIE,how='left',validate='one_to_one')
    clasificacion['en_poblacion_comun']=clasificacion.intervalo_ceros.notna()
    unido=cruzar(predicciones,clasificacion)
    unido['Tipo_demanda']=np.where(unido.y_real.eq(0),'cero','positiva')
    patrones=resumir(unido,'clasificacion_demanda',[x for x in PATRONES if x in set(unido.clasificacion_demanda) or x in PATRONES[:4]],'Clasificacion_demanda')
    grupos=resumir(unido,'grupo_demanda',[x for x in GRUPOS if x in set(unido.grupo_demanda) or x in GRUPOS[:2]])
    ceros=resumir(unido,'intervalo_ceros')
    ceros=ceros.merge(pd.DataFrame(bordes),on='intervalo_ceros',how='left',validate='many_to_one')
    tipos=resumir(unido,'Tipo_demanda',['cero','positiva'])
    negativas=auditoria_negativas(predicciones)
    serie=[]
    for (suc,prod,mod),d in unido.groupby(SERIE+['Modelo'],observed=True):
        serie.append({'sucursal':suc,'producto':prod,'Modelo':mod,**metricas(d),
            'clasificacion_demanda':d.clasificacion_demanda.iloc[0],
            'grupo_demanda':d.grupo_demanda.iloc[0],'porcentaje_ceros':d.porcentaje_ceros.iloc[0]})
    serie=pd.DataFrame(serie)
    minimos=conteo_minimos(serie)
    minimos_ml=conteo_minimos(serie.loc[serie.Modelo.isin(MODELOS[:3])],MODELOS[:3])
    contribuciones=[]
    for _,fila in tipos.iterrows():
        peso=fila.Observaciones/3444
        contribuciones.append({'Modelo':fila.Modelo,'Tipo_demanda':fila.Tipo_demanda,
            'Peso':peso,'Contribucion_MAE':peso*fila.MAE,'Contribucion_MSE':peso*fila.RMSE**2})
    estado_serie=referencia.groupby(SERIE,observed=True).y_real.max().gt(0).rename('Tiene_demanda_positiva_test').reset_index()
    minimos_tipo=[]
    extendida=serie.merge(estado_serie,on=SERIE,validate='many_to_one')
    for estado,d in extendida.groupby('Tiene_demanda_positiva_test'):
        conteos=conteo_minimos(d)
        conteos['Tipo_serie_test']='alguna_demanda_positiva' if estado else 'sólo_ceros'
        minimos_tipo.append(conteos)

    despliegue=[]
    for mod in MODELOS[:3]:
        glob=metricas(predicciones.loc[predicciones.Modelo.eq(mod)])
        item={'Modelo':mod,**{k+'_global':glob[k] for k in ['MAE','RMSE','R2']}}
        for grupo in GRUPOS[:2]:
            fila=grupos.loc[grupos.Modelo.eq(mod)&grupos.grupo_demanda.eq(grupo)].iloc[0]
            for met in ['MAE','RMSE']: item[met+'_'+grupo]=fila[met]
            item['Series_'+grupo]=int(fila.Series)
        pos=tipos.loc[tipos.Modelo.eq(mod)&tipos.Tipo_demanda.eq('positiva')].iloc[0]
        for met in ['MAE','RMSE']:item[met+'_demanda_positiva']=pos[met]
        neg=negativas.set_index('Modelo').loc[mod]
        for met in ['Predicciones_negativas','Porcentaje_predicciones_negativas']:item[met]=neg[met]
        cov=cobertura.set_index('Modelo').loc[mod]
        item.update(Series_cubiertas=int(cov.Series_evaluadas),Observaciones_cubiertas=int(cov.Observaciones_evaluadas),
            Observaciones_metricas_comunes=3444,Series_metricas_comunes=129,
            Criterio_cobertura='propia, sólo cobertura; todas las métricas son comunes')
        despliegue.append(item)
    distribucion=pd.DataFrame([{'clasificacion_demanda':p,'Series_comunes':int(c.clasificacion_demanda.eq(p).sum()),
        'Series_dataset':int(clasificacion.clasificacion_demanda.eq(p).sum())} for p in PATRONES])
    correl=[]
    for mod in MODELOS:
        d=serie.loc[serie.Modelo.eq(mod)]
        for met in ['MAE','RMSE']:
            # Pearson correlation of average ranks is Spearman, ties retained.
            valor=d.porcentaje_ceros.rank().corr(d[met].rank())
            correl.append({'Modelo':mod,'Metrica':met,'Spearman_por_serie':valor,'Series':len(d),
                'Interpretacion':'Asociación descriptiva; sin inferencia causal ni p-valor'})
    return {'clasificacion_intermitencia.csv':clasificacion,'metricas_por_intermitencia.csv':patrones,
        'metricas_por_grupo_demanda.csv':grupos,'metricas_por_ceros.csv':ceros,
        'metricas_cero_vs_positivo.csv':tipos,'auditoria_predicciones_negativas.csv':negativas,
        'metricas_por_serie.csv':serie,'conteo_minimos_por_serie.csv':minimos,
        'conteo_minimos_ml_por_serie.csv':minimos_ml,'conteo_minimos_por_tipo_serie.csv':pd.concat(minimos_tipo,ignore_index=True),
        'contribuciones_error.csv':pd.DataFrame(contribuciones),
        'evaluacion_despliegue.csv':pd.DataFrame(despliegue),'distribucion_intermitencia.csv':distribucion,
        'asociacion_ceros_error.csv':pd.DataFrame(correl)},unido,bordes


def ejecutar(root, generar_documentos=True):
    root=Path(root);salida=root/'resultados/semanal'
    # A fixed record of pre-analysis tracked artifacts, not a freshly invented baseline.
    baseline=json.loads((root/'docs/integridad_previa_intermitencia.json').read_text(encoding='utf-8'))
    for archivo,esperado in baseline['archivos'].items():
        assert huella(root/archivo)==esperado, f'Entrada protegida modificada: {archivo}'
    p=pd.read_csv(salida/'predicciones_comunes.csv',parse_dates=['semana'],float_precision='round_trip')
    s=pd.read_csv(salida/'dataset_semanal.csv',parse_dates=['semana'],float_precision='round_trip')
    cobertura=pd.read_csv(salida/'resumen_cobertura.csv',float_precision='round_trip')
    tablas,unido,bordes=calcular(s,p,cobertura)
    for nombre,d in tablas.items():d.to_csv(salida/nombre,index=False)
    # Existing aggregate metrics must remain exactly compatible.
    globales=pd.read_csv(salida/'comparacion_modelos_comun.csv').set_index('Modelo')
    for mod in MODELOS:
        m=metricas(p.loc[p.Modelo.eq(mod)])
        np.testing.assert_allclose([m[k] for k in ['MAE','RMSE','R2']],globales.loc[mod,['MAE','RMSE','R2']].to_numpy(dtype=float),rtol=1e-12,atol=1e-12)
    if generar_documentos:
        from src.informe_intermitencia import figuras, informe
        figuras(root,tablas)
        informe(root,tablas,unido,bordes)
    for archivo,esperado in baseline['archivos'].items():
        assert huella(root/archivo)==esperado, f'Se alteró evidencia: {archivo}'
    archivos=list(tablas)
    salida_hash={f'resultados/semanal/{nombre}':huella(salida/nombre) for nombre in archivos}
    if generar_documentos:
        for f in (salida/'figuras_intermitencia').glob('*.png'):
            salida_hash[f.relative_to(root).as_posix()]=huella(f)
        salida_hash['docs/ANALISIS_INTERMITENCIA.md']=huella(root/'docs/ANALISIS_INTERMITENCIA.md')
    manifiesto={'version_analisis':'intermitencia_train_adi_cv2_v1','commit_experimento_base':baseline['commit_base'],
        'entrenamiento_realizado':False,'predicciones_modificadas':False,'app_modificada':False,
        'observaciones_comunes':3444,'series_comunes':129,'modelos':MODELOS,'fecha_corte':str(CORTE.date()),
        'clasificacion':'sólo TRAIN; n/n_positivo; CV2 de tamaños positivos con varianza muestral ddof=1',
        'umbrales':{'ADI':ADI_CORTE,'CV2':CV2_CORTE,'igualdad':'lado superior >= corte'},
        'referencia':FUENTE,'intervalos_ceros':bordes,'empates_metricas':{'rtol':1e-9,'atol':1e-12},
        'R2':'NaN si n<2 o SST=0; no force_finite. Alertas relativas a varianza/ceros globales, no umbrales clínicos.',
        'python':platform.python_version(),'versiones':{n:importlib.metadata.version(n) for n in ['pandas','numpy','matplotlib']},
        'entradas_sha256':{f'resultados/semanal/{n}':huella(salida/n) for n in ['predicciones_comunes.csv','dataset_semanal.csv','resumen_cobertura.csv']},
        'codigo_sha256':{n:huella(root/n) for n in ['src/analisis_intermitencia.py','src/informe_intermitencia.py','scripts/analizar_intermitencia.py']},
        'salidas_sha256':salida_hash,'integridad_previa_sha256':huella(root/'docs/integridad_previa_intermitencia.json')}
    (salida/'experimento_intermitencia.json').write_text(json.dumps(manifiesto,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return tablas
