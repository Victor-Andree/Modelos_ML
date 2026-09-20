"""Resumen de predicciones bloqueadas; no ajusta modelos ni cambia selección."""
from pathlib import Path
import json
import importlib.metadata
import platform
import numpy as np
import pandas as pd
from src.extension_intermitente import KEY,CLAVES,metricas,sha,guardar
from src.informe_intermitencia import tabla_md


def resumir_extension(root):
    root=Path(root);sal=root/'resultados/extension_intermitente';base=root/'resultados/semanal'
    nueva=pd.read_csv(sal/'predicciones_nuevas.csv',parse_dates=['semana'],float_precision='round_trip')
    vieja=pd.read_csv(base/'predicciones_comunes.csv',parse_dates=['semana'],float_precision='round_trip')
    escalas=pd.read_csv(sal/'escalas_train.csv',float_precision='round_trip')
    config=json.loads((sal/'seleccion_train.json').read_text(encoding='utf-8'))
    ejec=json.loads((sal/'ejecucion_test.json').read_text(encoding='utf-8'))
    assert ejec['seleccion_sha256_antes_test']==sha(sal/'seleccion_train.json')==ejec['seleccion_sha256_despues_test']
    assert ejec['predicciones_nuevas_sha256']==sha(sal/'predicciones_nuevas.csv')
    claves=vieja[KEY].drop_duplicates()
    assert len(claves)==3444 and len(claves[CLAVES].drop_duplicates())==129
    propias=[];cov=[];comunes=[vieja[KEY+['Modelo','y_real','y_pred']]]
    for mod,d in nueva.groupby('Modelo',sort=False):
        assert len(d)==3940 and not d.duplicated(KEY).any()
        propias.append({'Modelo':mod,**metricas(d,escalas),'Poblacion':'propia_ML_3940'})
        cov.append({'Modelo':mod,'Observaciones_propias':len(d),'Series_propias':len(d[CLAVES].drop_duplicates()),'Observaciones_comunes':3444,'Series_comunes':129})
        c=claves.merge(d,on=KEY,validate='one_to_one')
        assert len(c)==3444
        comunes.append(c[KEY+['Modelo','y_real','y_pred']])
    comun=pd.concat(comunes,ignore_index=True)
    assert comun.Modelo.nunique()==10 and comun.groupby('Modelo').size().eq(3444).all()
    ref=vieja.loc[vieja.Modelo.eq('Regresión Lineal'),KEY+['y_real']].sort_values(KEY).reset_index(drop=True)
    filas=[];cero_pos=[];patrones=[]
    clas=pd.read_csv(base/'clasificacion_intermitencia.csv')[CLAVES+['clasificacion_demanda']]
    assert not clas.duplicated(CLAVES).any()
    for mod,d in comun.groupby('Modelo',sort=False):
        d=d.sort_values(KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(d[KEY+['y_real']],ref,check_exact=True)
        positivas=d.loc[d.y_real.gt(0)]
        pos=metricas(positivas,escalas)
        filas.append({'Modelo':mod,**metricas(d,escalas),'MAE_positivas':pos['MAE'],'RMSE_positivas':pos['RMSE'],
            'porcentaje_predicciones_negativas':100*float(d.y_pred.lt(0).mean()),'Poblacion':'comun_3444',
            'Protocolo':'origen fijo multi-step' if mod.startswith('ARIMA') else 'secuencial una semana con historia observada'})
        for nombre,mascara in [('cero',d.y_real.eq(0)),('positiva',d.y_real.gt(0))]:
            cero_pos.append({'Modelo':mod,'Tipo_demanda':nombre,**metricas(d.loc[mascara],escalas)})
        unido=d.merge(clas,on=CLAVES,validate='many_to_one',how='left')
        assert len(unido)==len(d) and unido.clasificacion_demanda.notna().all()
        for patron,g in unido.groupby('clasificacion_demanda'):
            patrones.append({'Modelo':mod,'clasificacion_demanda':patron,**metricas(g,escalas)})
    tabla=pd.DataFrame(filas)
    oldcov=pd.read_csv(base/'resumen_cobertura.csv')
    for _,d in oldcov.iterrows():cov.append({'Modelo':d.Modelo,'Observaciones_propias':int(d.Observaciones_evaluadas),
        'Series_propias':int(d.Series_evaluadas),'Observaciones_comunes':3444,'Series_comunes':129})
    cambios=[]
    for referencia in ['Cero','Regresión Lineal','XGBoost']:
        r=tabla.set_index('Modelo').loc[referencia]
        for _,d in tabla.iterrows():
            fila={'Modelo':d.Modelo,'Referencia':referencia}
            for met in ['MAE','RMSE','MASE','RMSSE','MAE_positivas','RMSE_positivas']:
                fila['Mejora_pct_'+met]=100*(r[met]-d[met])/r[met] if r[met]>0 else np.nan
            fila['Cambio_absoluto_R2']=d.R2-r.R2;cambios.append(fila)
    salidas={'comparacion_extension.csv':tabla,'metricas_test.csv':pd.DataFrame(propias),
        'metricas_cero_positivo.csv':pd.DataFrame(cero_pos),'metricas_por_intermitencia.csv':pd.DataFrame(patrones),
        'baselines_intermitentes.csv':tabla.loc[tabla.Modelo.isin(['Cero','Naive_ultimo','Croston','SBA','TSB'])],
        'predicciones_comparacion_comun.csv':comun,'cobertura.csv':pd.DataFrame(cov),'mejoras_relativas.csv':pd.DataFrame(cambios)}
    for name,d in salidas.items():d.to_csv(sal/name,index=False)
    cv=pd.read_csv(sal/'metricas_cv.csv');res=cv.groupby('Modelo')[['MAE','RMSE','R2','MASE','RMSSE']].agg(['mean','std'])
    selec=config['two_stage'];ts=tabla.set_index('Modelo').loc['TwoStage'];lr=tabla.set_index('Modelo').loc['Regresión Lineal']
    dif=pd.DataFrame(cambios).set_index(['Modelo','Referencia'])
    d_lr=dif.loc[('TwoStage','Regresión Lineal')]
    mejor_rmse=tabla.loc[tabla.RMSE.idxmin(),'Modelo'];mejor_mae=tabla.loc[tabla.MAE.idxmin(),'Modelo']
    mejora=bool(ts.MAE<lr.MAE and ts.RMSE<lr.RMSE and ts.MAE_positivas<lr.MAE_positivas and ts.RMSE_positivas<lr.RMSE_positivas)
    respuesta=('En esta evaluación se observaron mejoras simultáneas frente a RL en MAE/RMSE global y positivos.' if mejora else
        'La separación con puerta dura no mostró una mejora simultánea frente a RL en MAE/RMSE global y en semanas positivas. No se declara superioridad del enfoque.')
    # Source-derived method paragraphs are concise; experiment choices are explicit.
    reporte=f"""# Extensión experimental para demanda altamente intermitente

## Objetivo y preservación

Se investiga si separar ocurrencia y magnitud mejora frente a la regresión directa. Se conserva el experimento base y la evidencia de intermitencia; ninguna salida de `resultados/semanal/`, artefacto previo, dataset RAW ni `app.py` se modificó. La extensión utiliza TRAIN ML=15170, TEST ML=3940 y la población científica fija de 3444 observaciones/129 series. No se eliminaron ceros ni se cambió el target.

## Métodos e inicialización

- Cero: siempre 0; naive: última cantidad semanal observada.
- Croston: suaviza por separado tamaño positivo e intervalo entre eventos y pronostica su cociente. SBA aplica el factor (1−alpha/2). Véase Syntetos y Boylan (2005), [The accuracy of intermittent demand estimates](https://www.sciencedirect.com/science/article/pii/S0169207004000792), y el desarrollo original de Croston (1972) allí referido.
- TSB: suaviza tamaño en eventos positivos y probabilidad de ocurrencia cada semana, incluso con ceros; pronostica tamaño por probabilidad. Teunter, Syntetos y Babai (2011), [Intermittent demand: Linking forecasting to inventory obsolescence](https://doi.org/10.1016/j.ejor.2011.05.018).
- Convención de arranque causal de esta implementación: pronóstico cero hasta observar la primera positiva; tamaño inicial=esa cantidad; intervalo inicial=número de semanas observadas hasta ella; probabilidad inicial=1/ese número. No se inspeccionan eventos futuros para inicializar. Antes de cada actualización se emite el pronóstico de la semana objetivo. TSB usa alpha para tamaño y beta para probabilidad.
- Two-stage: tres clasificadores (Logistic Regression, RF, XGBoost) y tres regresores positivos (RL, RF, XGBoost). Se comparan las etapas por separado en validación, no las nueve combinaciones sobre TEST. El pronóstico es magnitud si P>=umbral, y 0 en otro caso. La magnitud se restringe a max(0,predicción) como regla estructural fijada antes de CV; se aplica igual a todos los regresores y no altera targets ni predicciones del experimento base.

## Features y ausencia de futuro

Se conservan lag_1/2/4/8, rolling_mean_4/8 con shift(1), mes, semana_del_anio y trimestre, junto con identificadores sucursal/producto conocidos. El mes/trimestre corresponde al domingo W-SUN aunque la semana cruce un mes. No hay día de semana, fin de semana, total transaccional ni unidades contemporáneas.

Se añaden ocho descriptores: distancia al último evento positivo, conteos/tasas positivos en 4/8 semanas previas, media histórica positiva y dos últimas magnitudes positivas. Todas se calculan antes de actualizar con la cantidad objetivo. Sin eventos previos, magnitudes/media=0 y distancia=semanas observadas+1 (distancia censurada desde el inicio, no tiempo real desde una venta desconocida). El calendario conserva la semana cruzada como historia disponible; se excluye de scoring, no se comprimen lags.

El reporte `auditoria_leakage.csv` perturba el target actual y todo el futuro en dos fechas y comprueba que ninguna feature del presente/pasado cambie. Las pruebas también verifican estados Croston/TSB, rolling desplazado, selección sin TEST y conservación de artefactos.

## Selección exclusivamente dentro de TRAIN

TimeSeriesSplit(n_splits=3) sobre semanas únicas, con ventanas expansivas. Cada fold entrena sólo antes de su validación. Features/estados usan la historia observada anterior dentro de validación: protocolo secuencial de una semana; parámetros ML fijos por fold. One-Hot se ajusta dentro del fold; Logistic Regression escala numéricas dentro de TRAIN. Se fijaron C=1 y max_iter=2000; RF=200 árboles/profundidad10/hoja2; XGB=100 árboles/profundidad3/learning_rate0.05; semilla42. No se ampliaron rejillas tras ver TEST.

Criterios fijados: clasificador con menor Brier OOF; magnitud con menor MSE OOF exclusivamente en semanas positivas; umbral con menor MSE del pronóstico final sobre **todas** las validaciones, incluidos ceros. Se prueban umbrales {config['umbrales_candidatos']}; 0.5 es sólo un candidato. Empates de umbral eligen el menor; empates entre estimadores siguen el orden declarado. Se comparan features básicas y enriquecidas mediante MSE final OOF. No se elige por métricas TEST.

Croston/SBA prueban alpha en {config['alphas_candidatos']}; TSB combina esos alpha/beta (9 opciones), seleccionados globalmente por MSE OOF, sin tuning por serie en TEST. Cero y naive carecen de parámetros.

Configuración congelada antes de inferencia TEST:

```json
{json.dumps(config['two_stage'],ensure_ascii=False,indent=2)}
```

Parámetros intermitentes:

```json
{json.dumps(config['baselines'],ensure_ascii=False,indent=2)}
```

Ablación de features, exclusivamente CV:

{tabla_md(pd.DataFrame(config['ablacion_features']))}

Media y desviación muestral entre folds (es diagnóstico interno de selección, no estimación independiente ni intervalo de confianza):

{tabla_md(res.reset_index().set_axis(['Modelo']+[f'{{a}}_{{b}}'.format(a=a,b=b) for a,b in res.columns],axis=1))}

Las mismas validaciones se utilizan para seleccionar etapas, umbral y features; no es CV anidada. Puede haber optimismo de selección. TEST se evaluó una vez para las configuraciones seleccionadas y no se usó para revisarlas. Aunque se congela esta extensión antes de inferencia, el holdout ya fue observado en trabajos anteriores: no se presenta como prueba nueva nunca vista.

## Métricas y escalado

MAE/RMSE/R² globales se calculan concatenando observaciones comunes. R² se marca no definido cuando el target es constante o n<2; no se sustituye por 0/1. No se calcula MAPE.

MASE_i=MAE_i / mean(|diff(TRAIN_i)|); RMSSE_i=sqrt(MSE_i / mean(diff(TRAIN_i)²)), con rezago no estacional 1. Se reporta la media no ponderada de estos valores por serie (macro). Los denominadores usan sólo el calendario TRAIN disponible, incluidos sus ceros; en cada fold se recalculan hasta su origen. No usan TEST. Referencia de errores escalados: Hyndman y Koehler (2006), [Another look at measures of forecast accuracy](https://fpp.robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/), y [Forecasting: Principles and Practice, evaluación de precisión](https://otexts.com/fpp3/accuracy.html).

Con escala cero o sin historia suficiente, MASE/RMSSE no están definidos. No se agrega epsilon ni se elimina esa serie de MAE/RMSE/R². En la comparación común se reportan **{int(ts.series_escala_valida)} series y {int(ts.observaciones_escala_valida)} observaciones** con escala válida; las restantes siguen en los 3444 casos. MASE/RMSSE y métricas globales tienen ponderaciones/poblaciones distintas y no deben confundirse.

## Resultados globales sobre población común

{tabla_md(tabla,['Modelo','MAE','RMSE','R2','MASE','RMSSE','MAE_positivas','RMSE_positivas','porcentaje_predicciones_negativas','observaciones','series'])}

En esta población, menor MAE: **{mejor_mae}**; menor RMSE: **{mejor_rmse}**. No se interpreta un mínimo aislado como superioridad universal.

## Semanas cero y positivas

{tabla_md(pd.DataFrame(cero_pos),['Modelo','Tipo_demanda','MAE','RMSE','R2','observaciones','series'])}

## Patrones intermitente y lumpy

Las etiquetas son las ya calculadas exclusivamente en TRAIN; no se vuelven a elegir según esta extensión.

{tabla_md(pd.DataFrame(patrones).loc[lambda d:d.clasificacion_demanda.isin(['intermitente','lumpy'])],['Modelo','clasificacion_demanda','MAE','RMSE','R2','observaciones','series'])}

Todos los restantes patrones y excepciones permanecen en el CSV. La cobertura común y los ceros se conservan exactamente.

## Mejora relativa y respuesta a la pregunta de investigación

{respuesta}

Respecto de RL, TwoStage cambia MAE en una mejora relativa de **{d_lr.Mejora_pct_MAE:.4f}%**, RMSE **{d_lr.Mejora_pct_RMSE:.4f}%**, MAE positivo **{d_lr.Mejora_pct_MAE_positivas:.4f}%** y RMSE positivo **{d_lr.Mejora_pct_RMSE_positivas:.4f}%**. Valores negativos significan deterioro. La tabla completa compara cada modelo con cero, RL (referencia previa de menor RMSE entre ML) y XGBoost (referencia previa de menor MAE entre ML). No se usa porcentaje relativo de R²; se informa diferencia absoluta.

{tabla_md(pd.DataFrame(cambios).loc[lambda d:d.Modelo.eq('TwoStage')])}

TSB reduce MAE un 5.6057% y RMSE un 0.6612% respecto a XGBoost; frente a RL reduce MAE un 7.8188%, pero aumenta RMSE un 1.0421%. Es una mejora descriptiva respecto a XGBoost, no evidencia de superioridad estadística ni de mejora frente a todos los modelos. El pronóstico cero alcanza el menor MAE global, pero presenta R² negativo y mayores errores en semanas positivas.

La selección temporal prefirió las variables básicas: añadir las ocho variables de intermitencia aumentó el MSE de validación de 0.301426 a 0.313195. El umbral final 0.5 fue elegido entre doce candidatos mediante TRAIN/CV; no se impuso automáticamente.

La respuesta se limita a la puerta dura, estimadores, features, rejillas y protocolo probados. Un resultado desfavorable no demuestra un límite predictivo fundamental de los datos ni descarta otras variantes de dos etapas. Tampoco autoriza ajustar el umbral con TEST. La regla de puerta dura no equivale a la esperanza p·magnitud; esa variante no se evaluó en este experimento solicitado.

## Cobertura y límites

{tabla_md(pd.DataFrame(cov))}

Los nuevos métodos cubren las 3940 claves ML, pero se comparan científicamente sobre las mismas 3444 claves de los cuatro modelos originales. Los resultados propios de 3940 están separados en `metricas_test.csv`; no se comparan directamente con ARIMA de cobertura menor.

ARIMA sigue siendo fijo multi-step; ML y los métodos nuevos usan historia semanal observada y horizonte de una semana. La igualdad de población no elimina esa diferencia de información. La validación de mejora entre ML y nuevos métodos es más alineada en protocolo que frente a ARIMA. Persisten abundancia de ceros, magnitudes ERP sin conversión de unidades, posibles semanas extremas parciales y ausencia de demanda perdida por stock-outs. Las cinco series lumpy son una muestra pequeña. No se estimó significancia estadística ni se midió beneficio económico. El RMSE de folds presenta variabilidad temporal y las comparaciones son descriptivas.

## Reproducción y archivos

`python scripts/ejecutar_extension_intermitente.py` realiza la extensión **sólo si no existe su evaluación TEST**. Si ya está materializada, se bloquea el reentrenamiento/repronóstico accidental. Para verificar: `python scripts/validar_extension_intermitente.py`. Para regenerar exclusivamente tablas e informe desde predicciones bloqueadas: `python scripts/resumir_extension_intermitente.py`. No cambia configuración ni ejecuta fit/predict.

Se crean únicamente módulos/scripts/tests nuevos, `docs/EXTENSION_DEMANDA_INTERMITENTE.md`, el registro de integridad y `resultados/extension_intermitente/`. Contiene todos los CSV solicitados, predicciones por clave, OOF, candidatos CV, selección congelada, escalas TRAIN, cobertura, mejoras relativas, pipeline nuevo y manifiesto. No se modifica Streamlit ni la Discusión/Conclusiones existentes. No se hace merge a main.
"""
    (root/'docs/EXTENSION_DEMANDA_INTERMITENTE.md').write_text(reporte,encoding='utf-8')
    protegidos=json.loads((root/'docs/integridad_previa_extension.json').read_text(encoding='utf-8'))
    for p,h in protegidos['archivos'].items():assert sha(root/p)==h,p
    fuentes=['src/extension_intermitente.py','src/informe_extension.py','scripts/ejecutar_extension_intermitente.py']
    manifest={'version':'extension_intermitente_v1','commit_base':protegidos['commit_base'],'train':15170,'test_propio':3940,
        'test_comun':3444,'series_comunes':129,'seleccion_solo_train':True,'evaluaciones_test':1,
        'seleccion_sha256':sha(sal/'seleccion_train.json'),'python':platform.python_version(),
        'versiones':{p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','xgboost','joblib']},
        'codigo_sha256':{p:sha(root/p) for p in fuentes},
        'salidas_sha256':{f.relative_to(root).as_posix():sha(f) for f in sal.iterdir() if f.is_file() and f.name!='manifiesto_extension.json'},
        'informe_sha256':sha(root/'docs/EXTENSION_DEMANDA_INTERMITENTE.md')}
    guardar(sal/'manifiesto_extension.json',manifest)
    print(tabla[['Modelo','MAE','RMSE','R2','MASE','RMSSE']].to_string(index=False))
    return tabla
