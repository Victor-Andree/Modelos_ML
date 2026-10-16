"""Consolida evidencia congelada y calcula Media_historica; no ajusta modelos."""
from pathlib import Path
import sys
import hashlib
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analizar_significancia_modelos import cargar_validar, ESPERADAS

KEY = ['sucursal', 'producto', 'semana']
SERIE = KEY[:2]
CORTE = pd.Timestamp('2026-01-01')
MODELOS = ['Regresión Lineal', 'Random Forest', 'XGBoost', 'ARIMA (Optimizado ADF/AIC)']
NOMBRES = {'Cero': 'Predictor cero', 'Naive_ultimo': 'Persistencia', 'Media_historica': 'Media histórica'}


def leer(path):
    d = pd.read_csv(path, float_precision='round_trip')
    if 'semana' in d:
        d['semana'] = pd.to_datetime(d.semana)
    return d


def huellas(root):
    # Todos los archivos experimentales y código originales, sin outputs propios.
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['resultados', 'datasets', 'src', 'notebooks']
            for p in (root/folder).rglob('*') if p.is_file()
            and 'anexo8' not in p.parts and '__pycache__' not in p.parts}


def media_historica(semanal, claves):
    """yhat_s,t = sum(y_s,u para u en TRAIN)/n_s, fija para todo TEST.

    TRAIN contiene semanas W-SUN completas anteriores al corte, incluidos ceros.
    No se usa el recorte de features ML ni se actualiza el promedio en TEST.
    """
    assert not semanal.duplicated(KEY).any()
    train = semanal.loc[(semanal.semana + pd.Timedelta(days=1)).le(CORTE)]
    assert train.cantidad.notna().all() and np.isfinite(train.cantidad).all()
    assert train.cantidad.ge(0).all() and train.semana.dt.dayofweek.eq(6).all()
    medias = train.groupby(SERIE, as_index=False).cantidad.mean().rename(columns={'cantidad':'media_train'})
    d = claves[KEY + ['y_real']].merge(medias, on=SERIE, how='left', validate='many_to_one')
    assert d.media_train.notna().all(), 'Series comunes sin historial TRAIN: detener, no imputar'
    assert not d.duplicated(KEY).any()
    assert (d.semana - pd.Timedelta(days=6)).ge(CORTE).all()
    d['y_pred'] = d.media_train
    return d[KEY + ['y_real', 'y_pred', 'media_train']]


def validar_claves(d, referencia):
    assert len(d) == 3444 and len(d[SERIE].drop_duplicates()) == 129
    assert not d.duplicated(KEY).any()
    a = d.set_index(KEY).sort_index()
    b = referencia.set_index(KEY).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_array_equal(a.y_real, b.y_real)


def metricas(d):
    """Errores semanales agregados, sin ajuste de estimadores."""
    error = d.y_real.to_numpy() - d.y_pred.to_numpy()
    sst = np.square(d.y_real - d.y_real.mean()).sum()
    assert len(d) > 1 and sst > 0
    assert np.isfinite(error).all()
    return {'MAE': float(np.abs(error).mean()),
            'RMSE': float(np.sqrt(np.square(error).mean())),
            'R2': float(1 - np.square(error).sum()/sst),
            'observaciones': len(d), 'series': len(d[SERIE].drop_duplicates())}


def resumen(nombre, d):
    m = metricas(d)
    assert np.isfinite([m[k] for k in ['MAE', 'RMSE', 'R2']]).all()
    por = d.assign(ae=(d.y_real-d.y_pred).abs()).groupby(SERIE).ae.mean()
    fila = dict(Modelo=nombre, **{k:m[k] for k in ['MAE','RMSE','R2']},
                MAE_medio_por_serie=por.mean(), DE_MAE_por_serie=por.std(ddof=1))
    fila.update(Observaciones=m['observaciones'], Series=m['series'])
    return fila


def preparar(root=ROOT):
    # Esta validación compara CSV por serie, predicciones TEST y métricas congeladas.
    cargar_validar(root)
    s = root/'resultados/semanal'
    comun = leer(s/'predicciones_comunes.csv')
    ref = comun.loc[comun.Modelo.eq(MODELOS[0]), KEY+['y_real']].copy()
    semanal = leer(s/'dataset_semanal.csv')
    train = semanal.loc[(semanal.semana+pd.Timedelta(days=1)).le(CORTE)]
    ml = leer(s/'dataset_train_ml.csv')
    assert len(ml)==15170 and (ml.semana+pd.Timedelta(days=1)).le(CORTE).all()
    control = ml.merge(train, on=KEY, suffixes=('_ml','_calendario'), validate='one_to_one')
    assert len(control)==len(ml)
    np.testing.assert_array_equal(control.cantidad_ml, control.cantidad_calendario)
    modelos = []
    for nombre in MODELOS:
        d = comun.loc[comun.Modelo.eq(nombre)]
        validar_claves(d, ref)
        f = resumen(nombre,d)
        np.testing.assert_allclose([f[k] for k in ['MAE','RMSE','R2']],ESPERADAS[nombre],rtol=0,atol=1e-12)
        modelos.append(f)
    propia = leer(s/'metricas_cobertura_propia.csv')
    assert set(propia.Modelo)==set(MODELOS) and len(propia)==4
    for row in propia.itertuples():
        nombre = 'predicciones_arima_concatenadas.csv' if row.Modelo.startswith('ARIMA') else 'predicciones_'+row.Modelo.replace(' ','_')+'_semanal.csv'
        m = metricas(leer(s/nombre))
        np.testing.assert_allclose([m[k] for k in ['MAE','RMSE','R2']],[row.MAE,row.RMSE,row.R2],rtol=0,atol=1e-12)
        assert (m['observaciones'],m['series']) == (row.Observaciones_evaluadas,row.Series_evaluadas)
        assert (m['observaciones'],m['series']) == ((3444,129) if row.Modelo.startswith('ARIMA') else (3940,156))
    existentes = leer(root/'resultados/extension_intermitente/predicciones_comparacion_comun.csv')
    bases = []
    for nombre in ['Cero','Naive_ultimo']:
        d = existentes.loc[existentes.Modelo.eq(nombre)]
        validar_claves(d,ref)
        f = resumen(NOMBRES[nombre],d)
        if nombre=='Cero':
            assert d.y_pred.eq(0).all()
        else:
            previo = semanal.sort_values(KEY).copy()
            previo['ultimo'] = previo.groupby(SERIE).cantidad.shift(1)
            control = d.merge(previo[KEY+['ultimo']],on=KEY,validate='one_to_one')
            assert len(control)==len(d)
            np.testing.assert_array_equal(control.y_pred,control.ultimo)
        bases.append(f)
    historica = media_historica(semanal,ref)
    validar_claves(historica,ref)
    bases.append(resumen('Media histórica',historica))
    return pd.DataFrame(modelos), pd.DataFrame(bases), propia, historica


def ejecutar(root=ROOT):
    antes = huellas(root)
    modelos,bases,propia,historica = preparar(root)  # Ninguna escritura hasta validar todo.
    out = root/'resultados/anexo8'
    out.mkdir(exist_ok=True)
    for nombre,d in [('modelos_poblacion_comun.csv',modelos),('lineas_base_poblacion_comun.csv',bases),('predicciones_media_historica.csv',historica)]:
        (out/nombre).write_bytes(d.to_csv(index=False,lineterminator='\n').encode('utf-8'))
    shutil.copyfile(root/'resultados/semanal/metricas_cobertura_propia.csv',out/'modelos_poblacion_propia.csv')
    tabla = pd.concat([modelos.assign(Tipo='Modelo'),bases.assign(Tipo='Línea base')],ignore_index=True)
    tabla['Modelo'] = tabla.Modelo.replace({'ARIMA (Optimizado ADF/AIC)':'ARIMA'})
    tabla = tabla.rename(columns={'Modelo':'Modelo/Referencia','DE_MAE_por_serie':'DE_MAE_serie'})
    tabla = tabla[['Modelo/Referencia','Tipo','MAE','RMSE','R2','DE_MAE_serie','Observaciones','Series']]
    (out/'tabla_anexo8.csv').write_bytes(tabla.to_csv(index=False,float_format='%.4f',lineterminator='\n').encode('utf-8'))
    assert antes==huellas(root), 'Se alteró evidencia previa'
    print(tabla.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    return modelos,bases,propia,historica


if __name__=='__main__':
    ejecutar()
