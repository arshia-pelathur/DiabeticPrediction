import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class preprocessor:
    def __init__(self,data):
        self.data = data
    
    def scale(self,data):
        mean = np.mean(data,axis=0)
        std_dev = np.std(data,axis=0)
        scaled_data = (data - mean ) / std_dev
        return scaled_data
    
    def preprocess(self):
        features = self.data.iloc[:,:-1]
        target = self.data.iloc[:,-1]
        X = np.array(self.scale(features))
        y = np.array(target)
        # X_ones = np.ones((X.shape[0],1))
        # X = np.concatenate((X,X_ones),axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

