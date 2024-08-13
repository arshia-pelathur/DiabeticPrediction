import pandas as pd

class load_file:
    def __init__(self,path):
        self.path = path
    def load(self):
        return pd.read_csv(self.path)