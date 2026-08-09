class DataSaver:
    
    def save_csv(self, dataframe, path):
        dataframe.to_csv(path, index=False)