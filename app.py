from pathlib import Path
import hashlib
import json
import joblib
import pandas as pd
import streamlit as st
from src.preprocessing.demanda_semanal import construir_semanal
from src.dashboard_semanal import entrada_pronostico

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Demanda farmacéutica semanal", layout="wide")
st.title("Demanda farmacéutica semanal")
st.caption("Regresión Lineal · sucursal-producto · semanas de lunes a domingo")

@st.cache_resource
def cargar_modelo(ruta, modificacion):
    return joblib.load(ruta)

@st.cache_data
def cargar_datos(ruta, modificacion):
    datos = pd.read_csv(ruta)
    datos['fecdoc'] = pd.to_datetime(datos['fecdoc'], errors='raise')
    return datos, construir_semanal(datos)

try:
    ruta = BASE / 'notebooks/modelo_regresion_lineal_semanal.pkl'
    meta = json.loads(ruta.with_suffix('.json').read_text(encoding='utf-8'))
    if hashlib.sha256(ruta.read_bytes()).hexdigest() != meta['sha256_modelo']:
        raise ValueError('El modelo no corresponde a sus metadatos. Exporte ambos desde EDA 07.')
    constructor = BASE / 'src/preprocessing/demanda_semanal.py'
    if hashlib.sha256(constructor.read_bytes()).hexdigest() != meta['sha256_constructor']:
        raise ValueError('El constructor semanal cambió desde la exportación. Verifique EDA 07.')
    limpieza = BASE / 'src/preprocessing/clean_data.py'
    if hashlib.sha256(limpieza.read_bytes()).hexdigest() != meta['sha256_limpieza']:
        raise ValueError('La limpieza cambió desde la exportación del modelo.')
    modelo = cargar_modelo(str(ruta), ruta.stat().st_mtime_ns)
    ruta_datos = BASE / 'datasets/dataset_maestro_dashboard.csv'
    df, semanal = cargar_datos(str(ruta_datos), ruta_datos.stat().st_mtime_ns)
except Exception as exc:
    st.error(f'No se pudo iniciar el dashboard: {exc}')
    st.stop()

sucursal = st.sidebar.selectbox('Sucursal', sorted(semanal.sucursal.unique()))
productos = sorted(semanal.loc[semanal.sucursal.eq(sucursal), 'producto'].unique())
producto = st.sidebar.selectbox('Producto', productos)
serie = semanal.loc[semanal.sucursal.eq(sucursal) & semanal.producto.eq(producto)]
st.info('Objetivo: cantidad semanal observada en ERP tras eliminar transacciones nulas y negativas antes de sumar. No incluye conversión entre cajas y unidades ni estima demanda perdida por falta de stock.')
a, b, c = st.columns(3)
a.metric('Cantidad semanal acumulada', f"{serie.cantidad.sum():,.2f}")
b.metric('Promedio por semana registrada', f"{serie.cantidad.mean():.3f}")
c.metric('Semanas con cantidad cero', f"{serie.cantidad.eq(0).mean():.1%}")
st.caption('Indicadores del calendario observado; los extremos pueden contener semanas parciales.')
st.line_chart(serie.set_index('semana')[['cantidad']], x_label='Semana (domingo)', y_label='Cantidad ERP')
st.subheader('Cantidad acumulada por sucursal para este producto')
st.bar_chart(semanal.loc[semanal.producto.eq(producto)].groupby('sucursal').cantidad.sum())

st.subheader('Pronóstico de una semana')
st.write('Indique hasta qué día está completa la carga del ERP. Se usan únicamente semanas cerradas; la última transacción por sí sola no demuestra que la carga esté completa.')
fecha = st.date_input('Datos completos hasta (inclusive)', value=df.fecdoc.max().date(),
                     min_value=df.fecdoc.min().date(), max_value=df.fecdoc.max().date())
confirmado = st.checkbox('Confirmo que la carga está completa hasta esa fecha')
st.caption(f"Modelo ajustado con TRAIN hasta {meta['train_fin'][:10]}. No se reentrena desde el dashboard.")
if confirmado:
    if pd.Timestamp(fecha) < pd.Timestamp(meta['train_fin']).normalize():
        st.warning('El origen debe ser posterior o igual al fin de TRAIN para evitar usar un modelo entrenado con datos futuros respecto a la consulta.')
    else:
        try:
            semana, entrada = entrada_pronostico(semanal, sucursal, producto, fecha)
            pred = float(modelo.predict(entrada[meta['columnas_entrada']])[0])
            st.metric('Predicción del modelo (cantidad ERP)', f'{pred:.4f}')
            st.write(f"Semana objetivo: {(semana - pd.Timedelta(days=6)):%d/%m/%Y} a {semana:%d/%m/%Y}.")
            if semana - pd.Timedelta(days=6) <= df.fecdoc.max().normalize():
                st.caption('La semana objetivo ya tiene fechas cubiertas por el archivo: esta consulta simula un origen histórico, no un pronóstico posterior a todo el dataset.')
            if pred < 0:
                st.warning('La regresión produjo un valor negativo. Se conserva para mantener el comportamiento evaluado; no constituye una cantidad a despachar.')
            st.caption('Estimación de demanda registrada. Para calcular pedidos se necesitan stock disponible, pedidos en tránsito, plazo de entrega y una política de servicio validada.')
            st.download_button('Descargar pronóstico', pd.DataFrame([{'sucursal': sucursal, 'producto': producto,
                'semana': semana, 'y_pred': pred, 'modelo': meta['modelo']}]).to_csv(index=False),
                file_name='pronostico_semanal.csv', mime='text/csv')
        except ValueError as exc:
            st.warning(str(exc))

st.subheader('Resultados del experimento temporal')
st.caption('Holdout desde 2026-01-01. El motor configurado es Regresión Lineal; la tabla común permite comparar las métricas actuales sin imponer un ganador.')
st.write('Métricas de cobertura propia del motor (población distinta de la comparación común):')
st.dataframe(pd.DataFrame([{'Modelo': meta['modelo'], **meta['metricas_holdout'],
    'Observaciones': meta['observaciones_holdout']}]), hide_index=True)
resultados = BASE / 'resultados/semanal'
try:
    ml = json.loads((resultados / 'experimento_ml.json').read_text(encoding='utf-8'))
    arima = json.loads((resultados / 'experimento_arima.json').read_text(encoding='utf-8'))
    for clave in ['sha256_dataset', 'sha256_constructor', 'version_target']:
        if not (ml[clave] == arima[clave] == meta[clave]):
            raise ValueError('Los resultados pertenecen a versiones distintas del experimento.')
    comun = json.loads((resultados / 'experimento_comun.json').read_text(encoding='utf-8'))
    for clave in ['sha256_dataset','sha256_constructor','sha256_limpieza','version_target']:
        if comun[clave] != meta[clave]:
            raise ValueError('La comparación común no corresponde al motor exportado.')
    st.write('Comparación sobre las mismas claves sucursal-producto-semana:')
    tabla_comun = pd.read_csv(resultados / 'comparacion_modelos_comun.csv')
    st.dataframe(tabla_comun.style.format({'MAE':'{:.4f}', 'RMSE':'{:.4f}', 'R2':'{:.4f}'}), hide_index=True)
    st.caption('La población de esta tabla es idéntica para los cuatro modelos. ARIMA usa origen fijo; ML usa historia observada semana a semana. No se declara un ganador general entre los cuatro modelos.')
except (OSError, ValueError, KeyError) as exc:
    st.info(f'Comparación de cuatro modelos no disponible: {exc}')
if hashlib.sha256(ruta_datos.read_bytes()).hexdigest() != meta['sha256_dataset']:
    st.caption('El archivo actual cambió respecto al entrenamiento. Las métricas mostradas corresponden al experimento original, no a una reevaluación de estos datos.')
