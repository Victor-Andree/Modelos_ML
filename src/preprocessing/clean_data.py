import pandas as pd

class DataCleaner: 
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.df.columns = self.df.columns.str.strip().str.lower()
        
    def extract_branch(self):
        self.df["sucursal"] = (
            self.df["numdoc"]
            .astype(str)
            .str.split("-")
            .str[0]
            .str.strip()
        )
        return self.df
         
    def remove_unused_columns(self):
        columns_to_remove = [
            "trab",
            "cliente",
            "doccli",
            "factorf",  
            "numdoc",   
            "tipago"    
        ]
        self.df.drop(
            columns=columns_to_remove,
            inplace=True,
            errors="ignore"
        )
        return self.df

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self.df
    
    def clean_text_columns(self):
        text_columns = self.df.select_dtypes(include="object").columns
        for column in text_columns:
            self.df[column] = (
                self.df[column]
                .astype(str)
                .str.strip()
            )
        return self.df
         
    def convert_datetime(self):
        self.df["fecdoc"] = pd.to_datetime(
            self.df["fecdoc"],
            errors="coerce"
        )
        return self.df
     
    def convert_numeric_columns(self):
        numeric_cols = ["cantidad"] 
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col], 
                    errors="coerce"
                )
        return self.df
    
    def remove_invalid_dates(self):
        self.df = self.df.dropna(subset=["fecdoc"])
        return self.df

    def filter_by_date(self):
        # Filtro estricto segun la metodologia (Enero 2024 - Julio 2026)
        fecha_inicio = '2024-01-01'
        fecha_fin = '2026-07-31'
        self.df = self.df[(self.df['fecdoc'] >= fecha_inicio) & (self.df['fecdoc'] <= fecha_fin)]
        return self.df
    
    def remove_invalid_quantity(self):
        self.df, self.auditoria_cantidad = depurar_demanda(self.df)
        return self.df
    
    def reset_index(self):
        self.df.reset_index(
            drop=True,
            inplace=True
        )
        return self.df
    
    def clean(self):
        self.extract_branch()
        self.remove_unused_columns()
        self.remove_duplicates()
        self.clean_text_columns()
        self.convert_datetime()
        self.remove_invalid_dates()
        self.filter_by_date()  # Se ejecuta el filtro de fechas aqui
        self.convert_numeric_columns()
        self.remove_invalid_quantity()
        self.reset_index()
        return self.df

def depurar_demanda(dataframe):
    """Copia analítica: elimina nulos y negativos sin alterar el original.

    No descarta ceros, no deduplica ni convierte unidades. Valores no numéricos
    o infinitos son errores de calidad, no ceros ni nulos silenciosos.
    """
    import numpy as np
    datos = dataframe.copy()
    datos['cantidad'] = pd.to_numeric(datos['cantidad'], errors='raise')
    nulos = datos.cantidad.isna()
    negativos = datos.cantidad.lt(0)
    if not np.isfinite(datos.loc[~nulos, 'cantidad']).all():
        raise ValueError('Cantidad contiene infinitos')
    eliminados = datos.loc[nulos | negativos]
    def afectados(marco, columna):
        return sorted(marco[columna].dropna().astype(str).unique().tolist()) if columna in marco else []
    auditoria = {
        'registros_iniciales': len(datos),
        'registros_negativos_eliminados': int(negativos.sum()),
        'suma_cantidades_negativas': float(datos.loc[negativos, 'cantidad'].sum()),
        'registros_nulos_eliminados': int(nulos.sum()),
        'registros_finales': int((~(nulos | negativos)).sum()),
        'productos_afectados': afectados(eliminados, 'producto'),
        'sucursales_afectadas': afectados(eliminados, 'sucursal'),
        'productos_afectados_negativos': afectados(datos.loc[negativos], 'producto'),
        'sucursales_afectadas_negativos': afectados(datos.loc[negativos], 'sucursal'),
        'productos_afectados_nulos': afectados(datos.loc[nulos], 'producto'),
        'sucursales_afectadas_nulos': afectados(datos.loc[nulos], 'sucursal'),
    }
    limpia = datos.loc[~(nulos | negativos)].reset_index(drop=True)
    assert limpia.cantidad.notna().all() and limpia.cantidad.ge(0).all()
    assert len(datos) == len(limpia) + int(negativos.sum()) + int(nulos.sum())
    return limpia, auditoria
