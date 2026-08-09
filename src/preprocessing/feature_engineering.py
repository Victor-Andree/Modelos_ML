import pandas as pd

class FeatureEngineer:
    def __init__(self,dataframe):
        self.df = dataframe.copy()
        
    def clean_branch(self):
        self.df["sucursal"] = (
            self.df["sucursal"]
            .astype(str)
            .str.replace(r'\D+', '', regex=True)
        )
    
    
        self.df["sucursal"] = pd.to_numeric(self.df["sucursal"], errors="coerce")
        return self.df
    
    
    def extract_time_features(self):
        self.df["mes"] = self.df["fecdoc"].dt.month
        
        
        def get_season(month):
            if month in [12, 1, 2]: return 'Verano'
            elif month in [3, 4, 5]: return 'Otoño'
            elif month in [6, 7, 8]: return 'Invierno'
            else: return 'Primavera'
            
            
        self.df["estacion"] = self.df["mes"].apply(get_season)
        return self.df
    
    def generate_features(self):
        self.clean_branch()
        self.extract_time_features()
        return self.df