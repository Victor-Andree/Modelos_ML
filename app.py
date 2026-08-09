import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Dashboard de Inteligencia Comercial - Farmacia")

# Carga de datos
@st.cache_data
def load_data():
    return pd.read_csv("datasets/final/AGUA OXIGENADA 10_final.csv")

df = load_data()

# Filtros laterales
st.sidebar.header("Filtros")
sucursal = st.sidebar.multiselect("Seleccionar Sucursal", options=df['sucursal'].unique())

# Filtro dinámico
if sucursal:
    df_filtered = df[df['sucursal'].isin(sucursal)]
else:
    df_filtered = df

# Visualizaciones
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demanda por Sucursal")
    fig1 = sns.barplot(data=df_filtered, x='sucursal', y='cantidad', estimator=sum)
    st.pyplot(fig1.figure)

with col2:
    st.subheader("Evolución Temporal")
    fig2 = sns.lineplot(data=df_filtered, x='fecdoc', y='cantidad')
    st.pyplot(fig2.figure)