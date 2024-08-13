import numpy as np
from sklearn.metrics import classification_report


class LogisticRegression:
    def __init__(self):
        pass

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def compute_weights(self,X,y):
        # Getting the transpose of X
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def fit(self,X,y):
        self.weights = self.compute_weights(X,y)

    def predict(self,X):
        self.z = X @ self.weights           # z = wx . Here x includes the bias in the matrix itself
        y_probability = self.sigmoid(self.z)
        return [1 if y >=0.5 else 0 for y in y_probability]