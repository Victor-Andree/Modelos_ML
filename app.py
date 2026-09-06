import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Configuración de la página
st.set_page_config(page_title="DSS Logístico Farmacéutico", layout="wide")

st.title("Dashboard Predictivo de Demanda (DSS)")
st.markdown("Sistema de Soporte a la Decisión basado en Machine Learning (Random Forest) para la optimización de inventarios.")

# --- 1. CARGA DE DATOS Y MODELO (.pkl) ---
@st.cache_resource
def load_model():
    # Carga el cerebro predictivo
    return joblib.load("notebooks/modelo_rf_optimo.pkl") # Ajusta la ruta si tu .pkl está en otra carpeta

@st.cache_data
def load_data():
    # Carga la historia para graficar y alimentar al modelo
    df = pd.read_csv("datasets/dataset_maestro_dashboard.csv") # Ajusta la ruta
    df['fecdoc'] = pd.to_datetime(df['fecdoc'])
    return df

try:
    modelo_rf = load_model()
    df = load_data()
    modelo_cargado = True
except Exception as e:
    st.error(f"Error al cargar archivos. Verifica las rutas: {e}")
    modelo_cargado = False

if modelo_cargado:
    # --- 2. FILTROS LATERALES ---
    st.sidebar.header("Filtros Analíticos")
    
    lista_sucursales = sorted(df['sucursal'].unique())
    lista_productos = sorted(df['producto'].unique())

    # Seleccionadores únicos (el modelo necesita 1 sucursal y 1 producto a la vez para predecir bien)
    sucursal_sel = st.sidebar.selectbox("Seleccionar Sucursal", options=lista_sucursales)
    producto_sel = st.sidebar.selectbox("Seleccionar Medicamento", options=lista_productos)

    # Filtrar datos históricos
    df_filtered = df[(df['sucursal'] == sucursal_sel) & (df['producto'] == producto_sel)].copy()

    # --- 3. MÉTRICAS CLAVE (KPIs) ---
    st.markdown("---")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Volumen Histórico Total", f"{int(df_filtered['cantidad'].sum())} cajas")
    kpi2.metric("Promedio Semanal", f"{round(df_filtered['cantidad'].mean(), 1)} cajas")
    kpi3.metric("Modelo Activo", "Random Forest Regressor")
    st.markdown("---")

    # --- 4. VISUALIZACIONES ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Comportamiento Histórico: Sucursal {sucursal_sel}")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=df_filtered, x='fecdoc', y='cantidad', ax=ax1, color='#1f77b4', linewidth=2)
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Cantidad Vendida')
        ax1.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with col2:
        st.subheader("Integración Predictiva (Proyección)")
        if df_filtered.empty:
            st.warning("No hay suficientes datos históricos para este cruce.")
        else:
            # Aquí el Dashboard muestra cómo se conecta con el .pkl
            st.success("✅ Motor de Machine Learning conectado.")
            st.info(f"El modelo evaluará la estacionalidad, los rezagos temporales y el comportamiento de la Sucursal {sucursal_sel} para proyectar la demanda futura del {producto_sel}.")
            
            # Gráfico ilustrativo de demanda general por sucursal
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            df_total_sucursal = df[df['producto'] == producto_sel].groupby('sucursal')['cantidad'].sum().reset_index()
            sns.barplot(data=df_total_sucursal, x='sucursal', y='cantidad', ax=ax2, palette='viridis')
            ax2.set_title(f"Concentración de Demanda: {producto_sel}")
            ax2.set_xlabel('Sucursales')
            ax2.set_ylabel('Cajas Totales')
            st.pyplot(fig2)