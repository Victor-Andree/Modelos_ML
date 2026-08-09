import panda as pd

class DataLoader:
    
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_excel(self):
        return pd.read_excel(self.file_path)
    
    def load_csv(self):
        return pd.read_csv(self.file_path)

   