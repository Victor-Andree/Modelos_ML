"""Tablas, figuras y borradores a partir de evidencia calculada; no entrenamiento."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analisis_intermitencia import MODELOS, PATRONES, FUENTE, metricas

CORTOS={MODELOS[0]:'Regresión Lineal', MODELOS[1]:'Random Forest', MODELOS[2]:'XGBoost', MODELOS[3]:'ARIMA'}
COLORES=['#0072B2','#E69F00','#009E73','#CC79A7']
ETIQUETAS={'suave':'Suave','erratica':'Errática','intermitente':'Intermitente','lumpy':'Lumpy',
    'positivas_insuficientes':'Una semana\npositiva en TRAIN',
    'sin_demanda_positiva':'Sin positivas\nen TRAIN','sin_historia_train':'Sin historia\nTRAIN'}


def figuras(root,t):
    destino=Path(root)/'resultados/semanal/figuras_intermitencia'
    destino.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    def guardar(fig,nombre):
        fig.savefig(destino/nombre,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    patrones=t['metricas_por_intermitencia.csv']
    cats=patrones.Clasificacion_demanda.drop_duplicates().tolist()
    for met,nombre in [('MAE','01_mae_por_patron.png'),('RMSE','02_rmse_por_patron.png')]:
        fig,ax=plt.subplots(figsize=(12,5.5),layout='constrained')
        x=np.arange(len(cats));ancho=.19
        for j,mod in enumerate(MODELOS):
            d=patrones.loc[patrones.Modelo.eq(mod)].set_index('Clasificacion_demanda').reindex(cats)
            ax.bar(x+(j-1.5)*ancho,d[met],ancho,label=CORTOS[mod],color=COLORES[j])
        cuentas=patrones.loc[patrones.Modelo.eq(MODELOS[0])].set_index('Clasificacion_demanda').Series
        ax.set_xticks(x,[ETIQUETAS[c]+f'\n(n={cuentas[c]} series)' for c in cats])
        ax.set_ylabel(f'{met} · cantidad ERP semanal')
        ax.set_title(f'{met} por patrón definido en TRAIN · población común')
        ax.legend(ncols=4,loc='upper center',bbox_to_anchor=(.5,1.0),frameon=False)
        ax.set_ylim(0,patrones[met].max()*1.3)
        ax.yaxis.grid(True,alpha=.2);ax.set_axisbelow(True)
        for i,c in enumerate(cats):
            if cuentas[c]==0:ax.text(i,.025,'Sin datos',ha='center',va='bottom',fontsize=9)
        guardar(fig,nombre)
    ceros=t['metricas_por_ceros.csv'];cats=ceros.intervalo_ceros.drop_duplicates().tolist()
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    labels=[]
    for q in cats:
        d=ceros.loc[ceros.intervalo_ceros.eq(q)].iloc[0]
        labels.append(f'{q}: {d.limite_inferior:.2f}–{d.limite_superior:.2f}%\n{int(d.Series)} series / {int(d.Observaciones)} semanas')
    for ax,met in zip(axes,['MAE','RMSE']):
        for j,mod in enumerate(MODELOS):
            d=ceros.loc[ceros.Modelo.eq(mod)].set_index('intervalo_ceros').reindex(cats)
            ax.plot(np.arange(len(cats)),d[met],marker='o',label=CORTOS[mod],color=COLORES[j])
        ax.set_xticks(np.arange(len(cats)),labels,fontsize=8)
        ax.set_title(met);ax.set_ylabel('Cantidad ERP semanal');ax.grid(alpha=.2)
    axes[0].legend(fontsize=9,frameon=False)
    fig.suptitle('Error y porcentaje de ceros en TRAIN · cuartiles con cortes repetidos colapsados')
    guardar(fig,'03_error_por_ceros.png')
    tipos=t['metricas_cero_vs_positivo.csv']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,met in zip(axes,['MAE','RMSE']):
        for j,mod in enumerate(MODELOS):
            d=tipos.loc[tipos.Modelo.eq(mod)].set_index('Tipo_demanda').reindex(['cero','positiva'])
            ax.bar(np.arange(2)+(j-1.5)*.19,d[met],.19,color=COLORES[j],label=CORTOS[mod])
        ax.set_xticks([0,1],['Demanda cero\n3 019 observaciones','Demanda positiva\n425 observaciones'])
        ax.set_title(met);ax.set_ylabel('Cantidad ERP semanal');ax.yaxis.grid(alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=9,frameon=False)
    fig.suptitle('Errores condicionados al valor real en TEST · mismos casos para cuatro modelos')
    guardar(fig,'04_error_cero_vs_positivo.png')
    distrib=t['distribucion_intermitencia.csv'].set_index('clasificacion_demanda')
    cats=[p for p in PATRONES if p!='sin_historia_train']
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    vals=distrib.loc[cats,'Series_comunes']
    barras=ax.bar(np.arange(len(cats)),vals,color=['#56B4E9']*4+['#999999']*2)
    ax.bar_label(barras,padding=3)
    ax.set_xticks(np.arange(len(cats)),[ETIQUETAS[c] for c in cats])
    ax.set_ylim(0,max(vals)*1.2);ax.set_ylabel('Número de series sucursal-producto')
    ax.set_title('Distribución de las 129 series comunes · clasificación con TRAIN')
    guardar(fig,'05_distribucion_patrones.png')


def tabla_md(d,columnas=None):
    if columnas is not None:d=d[columnas]
    def f(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
                      ['| '+' | '.join(f(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])


def informe(root,t,unido,bordes):
    root=Path(root);s=root/'resultados/semanal'
    pat=t['metricas_por_intermitencia.csv'];gr=t['metricas_por_grupo_demanda.csv']
    tipos=t['metricas_cero_vs_positivo.csv'];neg=t['auditoria_predicciones_negativas.csv']
    clas=t['clasificacion_intermitencia.csv'];c=clas.loc[clas.en_poblacion_comun]
    ref=unido.loc[unido.Modelo.eq(MODELOS[0])]
    cero=ref.y_real.eq(0);pct=100*cero.mean()
    glob=pd.read_csv(s/'comparacion_modelos_comun.csv')
    cov=pd.read_csv(s/'resumen_cobertura.csv')
    pos=tipos.loc[tipos.Tipo_demanda.eq('positiva')].set_index('Modelo')
    cer=tipos.loc[tipos.Tipo_demanda.eq('cero')].set_index('Modelo')
    cont=t['contribuciones_error.csv'].set_index(['Modelo','Tipo_demanda'])
    delta0=cont.loc[('XGBoost','cero'),'Contribucion_MAE']-cont.loc[('Regresión Lineal','cero'),'Contribucion_MAE']
    deltap=cont.loc[('XGBoost','positiva'),'Contribucion_MAE']-cont.loc[('Regresión Lineal','positiva'),'Contribucion_MAE']
    total_train=int(c.n_semanas.sum());positivas_train=int(c.n_semanas_demanda_positiva.sum())
    fr_train=100*(1-positivas_train/total_train)
    distrib=t['distribucion_intermitencia.csv']
    dpat=tabla_md(pat,['Modelo','Clasificacion_demanda','MAE','RMSE','R2','Observaciones','Series'])
    report=f"""# Análisis de intermitencia y evidencia para despliegue

Este análisis adicional usa exclusivamente las predicciones comunes existentes. No se entrenaron modelos, no se alteraron hiperparámetros, target, prueba ni predicciones y no se modificó `app.py`. Los borradores del final no sustituyen la Discusión ni las Conclusiones existentes del artículo.

## A. Metodología y referencia

El EDA anterior sólo contaba semanas cero y demanda promedio; sus tablas completa/evaluable no definían categorías ADI/CV². Se conserva esa evidencia. La clasificación nueva usa sólo semanas enteramente anteriores a 2026-01-01: `semana + 1 día <= corte`. No consulta valores del test y mantiene la historia anterior a la eliminación de lags. La semana cruzada no participa en la clasificación.

Se adopta la separación descriptiva ADI=1.32 y CV²=0.49 de Syntetos, Boylan y Croston (2005), *On the categorization of demand patterns*, Journal of the Operational Research Society, 56, 495–503, [DOI]({FUENTE}), figura 3. El trabajo fundamenta estos cortes para otros métodos de pronóstico; aquí sirven como taxonomía, no como garantía de superioridad para RL/RF/XGBoost/ARIMA.

Estimadores explícitos: ADI=N/N+ (semanas observadas / semanas positivas, incluyendo ceros iniciales/finales del calendario disponible); CV²=(s+/media+)², con s+ muestral (`ddof=1`) calculada únicamente sobre tamaños positivos. Esta elección de estimadores, tratamiento de límites y casos insuficientes es una convención reproducible del presente análisis; no se atribuye al artículo una regla para casos no estimables.

| Patrón | ADI | CV² |
|---|---|---|
| Suave | <1.32 | <0.49 |
| Errática | <1.32 | >=0.49 |
| Intermitente | >=1.32 | <0.49 |
| Lumpy | >=1.32 | >=0.49 |

La igualdad se asigna al lado superior. `grupo_demanda` usa solamente ADI: menor intermitencia (<1.32), mayor intermitencia (>=1.32). No se divide por desempeño. Con una positiva se estima ADI y el grupo, pero CV² no es estimable y se etiqueta `positivas_insuficientes`. Con ninguna positiva, ADI/CV² quedan ausentes y se conserva el grupo separado `sin_demanda_positiva`. Sin historia TRAIN se marca `sin_historia_train`. No se elimina ninguna observación por estas excepciones.

## B. Distribución y posibilidad real de contraste

{tabla_md(distrib)}

En la población común hay 1 serie suave, 0 erráticas, 63 intermitentes, 5 lumpy, 23 con una sola positiva y 37 sin positivas en TRAIN. La comparación binaria incluye 1 serie de menor intermitencia y 91 de mayor intermitencia; las 37 sin positivas permanecen aparte. Por tanto, no hay evidencia suficiente para generalizar diferencias entre demanda continua y demanda intermitente: el primer grupo tiene sólo una serie (30 observaciones de TEST). Las 5 lumpy aportan 104 observaciones.

## C. Ceros y estabilidad de las etiquetas

En TRAIN de las series comunes hay {total_train} semanas, {positivas_train} positivas y {fr_train:.4f}% ceros (ponderado por semanas). El promedio de los porcentajes por serie es {c.porcentaje_ceros.mean():.4f}%, y la mediana es {c.porcentaje_ceros.median():.4f}%. Son denominadores distintos y no deben intercambiarse.

En TEST común: **{int(cero.sum())} ceros de {len(ref)} ({pct:.4f}%)** y **{int((~cero).sum())} positivas ({100-pct:.4f}%)**. Existen 63 series con sólo ceros en TEST y 66 con alguna positiva. Una etiqueta sin positivas en TRAIN no asegura ausencia de demanda futura: ese grupo contiene 7 semanas positivas en TEST.

Los intervalos de ceros derivan de cuartiles del porcentaje TRAIN entre las 129 series comunes, con cada serie ponderada una vez. Se colapsan los cortes duplicados, obteniendo tres intervalos; el primero incluye ambos extremos y los restantes son abiertos a izquierda/cerrados a derecha. No se usan errores ni métricas para definirlos.

{tabla_md(pd.DataFrame(bordes))}

{tabla_md(t['metricas_por_ceros.csv'],['Modelo','intervalo_ceros','MAE','RMSE','R2','Observaciones','Series'])}

{tabla_md(t['asociacion_ceros_error.csv'],['Modelo','Metrica','Spearman_por_serie','Series'])}

Las correlaciones son descriptivas entre porcentaje de ceros TRAIN y error TEST por serie. No se calculan p-valores ni se afirma causalidad; escala de demanda, producto, sucursal, longitud de serie y régimen temporal pueden influir conjuntamente.

## D. Métricas por patrón y grupo

{dpat}

{tabla_md(gr,['Modelo','grupo_demanda','MAE','RMSE','R2','Observaciones','Series'])}

Se observó que XGBoost presenta menor MAE en las 63 series intermitentes y RL menor RMSE en ese patrón; RF presenta menor MAE/RMSE en las 5 lumpy. El grupo suave favorece RL en ambas métricas, pero contiene una única serie. No se generaliza ese resultado a demanda continua en toda la red.

R² se conserva incluso cuando es negativo. Se informa `Varianza_y_real`, SST, porcentaje de ceros, estado y advertencias. Las alertas señalan varianza inferior a la global o ceros superiores al porcentaje global; son comparaciones descriptivas, no umbrales de inestabilidad universal. Los grupos con una sola positiva o ninguna positiva en TRAIN muestran R² próximo a cero/negativo y baja varianza TEST. Cuando SST=0 o n<2, R² no está definido y se escribe vacío/NaN con motivo, nunca un 0 o 1 artificial. La categoría errática se muestra sin datos, no con error cero.

## E. Errores sobre semanas con demanda cero

{tabla_md(tipos.loc[tipos.Tipo_demanda.eq('cero')],['Modelo','MAE','RMSE','Observaciones','Series','R2_estado'])}

R² no se usa aquí porque y_real es constante. XGBoost obtiene el menor MAE; RL obtiene el menor RMSE en este subconjunto.

## F. Errores sobre semanas con demanda positiva

{tabla_md(tipos.loc[tipos.Tipo_demanda.eq('positiva')],['Modelo','MAE','RMSE','R2','Observaciones','Series'])}

En 425 semanas positivas de 66 series, RL presenta el menor MAE ({pos.loc['Regresión Lineal','MAE']:.4f}), muy cercano al de RF ({pos.loc['Random Forest','MAE']:.4f}); RF presenta el menor RMSE ({pos.loc['Random Forest','RMSE']:.4f}). XGBoost obtiene MAE {pos.loc['XGBoost','MAE']:.4f} y RMSE {pos.loc['XGBoost','RMSE']:.4f}. Todos los R² condicionados a positivas son negativos: respecto de la media de este subconjunto como referencia descriptiva, los errores cuadrados son mayores. Esa media no es un pronosticador disponible a priori y este análisis condicionado al resultado no sustituye una evaluación prospectiva.

## G. Predicciones negativas sin recorte

{tabla_md(neg)}

El criterio es estrictamente y_pred<0, sin tolerancia ni clip. RL tiene 684 negativas (676 con y_real=0, 8 con y_real>0), mínimo −0.055221. ARIMA tiene 1048 (1038/10), mínimo −0.262526; la mediana y percentil 95 de su magnitud son aproximadamente 0.000005, de modo que el conteo de signos por sí solo oculta magnitudes muy distintas. RF y XGBoost no presentan negativas en esta evaluación; ello no constituye una garantía universal para cualquier dato o versión. Un eventual recorte requiere una evaluación separada y no se aplicó.

## H. Desempeño por serie y cobertura

Mínimos entre los cuatro modelos:

{tabla_md(t['conteo_minimos_por_serie.csv'])}

Se reconocen empates con `rtol=1e-9, atol=1e-12`; se conservan las métricas originales. No hubo empates en estos conteos. El número de mínimos da el mismo peso a series con magnitudes y longitudes distintas; no equivale al MAE/RMSE agregado por observación ni implica superioridad global.

{tabla_md(t['conteo_minimos_por_tipo_serie.csv'])}

La separación entre series con sólo ceros y con alguna positiva evita interpretar automáticamente muchos mínimos de ARIMA como ventaja sobre semanas de demanda efectiva.

Cobertura propia, **sin mezclar métricas**:

{tabla_md(cov,['Modelo','Observaciones_evaluadas','Series_evaluadas','Observaciones_comunes','Series_comunes','Observaciones_fuera_comun'])}

En un sistema multisucursal, los tres ML evaluaron 3940 observaciones de 156 series y ARIMA 3444 de 129. Las 496 observaciones/27 series adicionales de ML indican aplicabilidad histórica bajo los filtros vigentes, no precisión adicional demostrada ni garantía de cobertura futura. Todas las métricas del presente informe usan sólo la población común. ARIMA permanece como comparación científica, no como motor preseleccionado del dashboard.

## I. Comparación de candidatos operativos

{tabla_md(t['evaluacion_despliegue.csv'])}

Mínimos entre **sólo los tres ML**:

{tabla_md(t['conteo_minimos_ml_por_serie.csv'])}

No hay puntajes, pesos de negocio inventados ni ranking por complejidad. Las pequeñas diferencias numéricas no se presentan como estadísticamente significativas.

## J. Por qué el promedio global puede ocultar las semanas positivas

El MAE global es p0·MAE0 + p+·MAE+. El MSE global tiene la misma descomposición con RMSE²; no se promedian directamente los RMSE. Los pesos son {cero.mean():.6f} y {1-cero.mean():.6f}. Las contribuciones exactas están en `contribuciones_error.csv`.

Para XGBoost menos RL, el componente de ceros cambia el MAE global en {delta0:+.6f}; el componente positivo lo cambia en {deltap:+.6f}. El total es {delta0+deltap:+.6f}: su ventaja de MAE global se explica aritméticamente por la reducción de error absoluto en ceros, a pesar de un MAE mayor en positivas. Esto describe esta muestra, no un mecanismo causal del algoritmo.

RL tiene menor error cuadrático agregado que XGBoost tanto en semanas cero como positivas; por eso mantiene menor RMSE y mayor R² global frente a XGBoost. RF mejora RMSE en positivas respecto de RL, pero su error cuadrático sobre ceros es mayor y el balance global favorece RL. En la misma población, el orden por R² y RMSE no aporta dos evidencias independientes, pues ambos dependen de la misma suma de errores cuadrados.

## K. Limitaciones y reproducibilidad

- Clasificación fija usando TRAIN; puede cambiar el régimen en TEST. Sólo una serie suave, ninguna errática y 60/129 series sin CV² estimable. No se fuerza un contraste equilibrado.
- ADI se estima como N/N+, no como media de distancias entre eventos descartando extremos; CV² usa ddof=1. Cerca del umbral o con pocas positivas, otras convenciones podrían cambiar la etiqueta. No se afinó la convención en función de los errores.
- No se usan periodos futuros para clasificar. La selección previa de la población común depende de cobertura/éxito de los modelos; es una población condicionada.
- ARIMA fijo multi-step y ML secuencial conservan protocolos distintos. La población común no permite atribuir diferencias sólo al algoritmo.
- Semanas extremas potencialmente parciales y ceros interiores bajo continuidad del ERP; cantidad observada no equivale a demanda latente ni mezcla cajas con unidades. Eliminar negativos conserva la definición del experimento base.
- Métricas microponderadas por observación, dependencia temporal y entre series, sin intervalos de confianza ni contrastes estadísticos. Correlaciones y conteos no establecen causalidad.
- La evaluación de negativas no modifica las predicciones. Una decisión operativa necesita definir costos, nivel de servicio, inventario y horizonte; no se inventaron esos criterios.
- Análisis posterior al holdout: una selección resultante requiere validación futura independiente.

Reproducir sin entrenar: `python scripts/analizar_intermitencia.py`; pruebas: `python -m unittest discover -p "test_*.py"`; verificaciones: `python scripts/validar_analisis_intermitencia.py` y `python scripts/validar_evidencia_semanal.py`. El notebook 08 usa este mismo análisis y no llama al runner de entrenamiento. `experimento_intermitencia.json` registra método, versiones, umbrales, cortes, hashes y ausencia de entrenamiento. `docs/integridad_previa_intermitencia.json` protege los archivos del experimento base, incluidos modelos, predicciones, RAW, resultados históricos y app.py. Las figuras PNG son de 300 dpi y muestran series/casos de cada subconjunto.

## Archivos generados y modificados

Se actualizó únicamente el notebook 08 y los atributos de checkout entre los archivos previos permitidos. Se añadieron `src/analisis_intermitencia.py`, `src/informe_intermitencia.py`, `scripts/analizar_intermitencia.py`, `scripts/validar_analisis_intermitencia.py` y `test_analisis_intermitencia.py`. La evidencia anterior queda protegida por `docs/integridad_previa_intermitencia.json`.

En `resultados/semanal/` se generaron:

{chr(10).join('- `' + nombre + '`' for nombre in t)}

Además, `experimento_intermitencia.json`, cinco PNG de 300 dpi en `figuras_intermitencia/` y este informe automático. Se conserva el experimento previo sin sobrescribirlo. No se hizo merge a main.

## Borrador para Resultados

Se analizaron 3444 observaciones comunes de 129 series sucursal-producto. Según TRAIN, se identificaron 1 serie suave, 63 intermitentes y 5 lumpy; no se observaron erráticas. Otras 23 series tuvieron una sola semana positiva y 37 ninguna, por lo que no se estimó CV². El test incluyó 3019 semanas cero ({pct:.2f}%) y 425 positivas. En las series intermitentes, XGBoost registró MAE 0.3727 y RL RMSE 0.7201; en lumpy, RF obtuvo MAE 0.4589 y RMSE 0.7318. En las semanas positivas, los MAE de RL, RF, XGBoost y ARIMA fueron 0.9787, 0.9794, 1.0007 y 1.0306; los RMSE fueron 1.3638, 1.3479, 1.3692 y 1.3931. Se registraron 684 predicciones negativas de RL y 1048 de ARIMA, frente a ninguna de RF/XGBoost. Las métricas globales comunes permanecieron inalteradas.

## Borrador para Discusión

En el conjunto evaluado, el menor MAE global de XGBoost coexistió con mayor MAE en semanas positivas que RL y RF. La descomposición del error sugiere que la frecuencia de ceros y su menor error absoluto en esas semanas explican aritméticamente la ventaja global. RL obtuvo menor RMSE y mayor R² global, mientras RF mostró menor RMSE cuando existió demanda positiva y mejor MAE/RMSE en las cinco series lumpy. Estos resultados sugieren un comportamiento dependiente del patrón y del criterio de error, sin sustentar superioridad universal. La taxonomía ADI/CV² se apoya en Syntetos et al. (2005); su transferencia a este contexto sirve como descripción y no como regla automática de selección. Cualquier explicación general sobre mecanismos de aprendizaje de los algoritmos ante ceros requiere validación externa [CITA NECESARIA]. La categoría de menor intermitencia cuenta con una sola serie, lo cual impide generalizar el contraste con demanda continua. ARIMA obtuvo numerosos mínimos por serie, pero ello debe interpretarse junto con la abundancia de series de cero, su menor cobertura y su protocolo de origen fijo. ML cubrió 27 series adicionales, pero esa cobertura no prueba mayor exactitud. El análisis es descriptivo, condicionado a la población común y a un holdout ya observado.

## Borrador para Conclusiones

La comparación de RL, RF, XGBoost y ARIMA sobre las mismas observaciones mostró fortalezas complementarias. XGBoost conservó el menor MAE global y RL el menor RMSE/mayor R²; RF presentó menor RMSE en semanas positivas y menor MAE/RMSE en el reducido grupo lumpy. ARIMA mantuvo utilidad como referencia científica, con diferencias de protocolo y cobertura que limitan la atribución de resultados al algoritmo. La elevada proporción de ceros y la escasez de patrones no intermitentes restringen la generalización. La conclusión científica es que ningún modelo domina simultáneamente todos los criterios estudiados. La decisión de implementación queda separada y requiere priorizar objetivos operativos y confirmar su desempeño prospectivamente.

## Evidence for deployment decision

**Recomendación técnica: no sustituir todavía el motor de Streamlit.** Evaluar posteriormente el siguiente compromiso, con los tres ML sobre las mismas observaciones y con la misma cobertura propia:

- **RL:** menor RMSE global (0.5396), mayor R² (0.3219), menor MAE en positivas (0.9787) y menor RMSE en intermitentes; genera 684 negativas (19.8606%).
- **RF:** cero negativas observadas, menor RMSE en positivas (1.3479), mejor MAE/RMSE en lumpy y MAE positivo cercano al de RL. Es un candidato operativo razonable si se priorizan errores grandes en demanda efectiva y salidas no negativas, pero no domina el MAE global ni el RMSE global.
- **XGBoost:** menor MAE global (0.2252), en ceros (0.1160) y en el grupo de mayor intermitencia (0.2972), sin negativas observadas; MAE/RMSE en positivas mayores que RL y RF.

No se declara un ganador operativo automático. La decisión depende de una prioridad de negocio aún no fijada; no se creó un puntaje compuesto y `app.py` permanece intacto.
"""
    (root/'docs/ANALISIS_INTERMITENCIA.md').write_text(report,encoding='utf-8')
