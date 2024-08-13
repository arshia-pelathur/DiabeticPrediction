import numpy as np
from logistic_regression import LogisticRegression

class GradientDescent(LogisticRegression):
    def __init__(self,lr,iterations):
        self.iterations = iterations
        self.alpha = lr

    def gradients(self, X, y, w, b):
        self.z = np.dot(X[:,:-1],w)  +  b          # z = wx + b
        y_pred = self.sigmoid(self.z)
        djdw = (1/X.shape[0]) * np.dot(X[:,:-1].T,(y_pred - y))                 # (1/m) * (f_wb - y)x
        djdb = (1/X.shape[0]) * np.sum((y_pred - y))
        return djdw, djdb

    def fit(self,X,y):
        self.weights = np.zeros(X.shape[1] - 1)      # matrix of 0s with shape n-1 to remove the last columns for the intercept
        self.bias = 0

        for _ in range(self.iterations):
            djdw , djdb = self.gradients(X, y, self.weights, self.bias)
            self.weights = self.weights - self.alpha * djdw
            self.bias = self.bias - self.alpha * djdb
        


    def predict(self,X):
        # Fitting the model with final weights, final bias
        self.z = np.dot(X[:,:-1],self.weights)  +  self.bias          # z = wx + b
        y_probability = self.sigmoid(self.z)
        return [1 if y >=0.5 else 0 for y in y_probability]
