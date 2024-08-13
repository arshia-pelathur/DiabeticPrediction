from gradient_descent import GradientDescent
import numpy as np


class LogisticGradientRegularization(GradientDescent):
    def __init__ (self, 
                  lr = 0.001, 
                  iterations = 1000, 
                  regularization = 'l2', 
                  lambda_ = 10):
        super().__init__(lr,iterations)
        self.regularization = regularization
        self.lambda_ = lambda_

    def regularize(self,X,type):
        if type == 'l2':
            dw = (self.lambda_ /X.shape[0]) * self.weights
        else:
            dw = (self.lambda_/X.shape[0]) * np.sign(self.weights)
        return dw



    def compute_gradients(self, X, y):
        w = self.weights
        b = self.bias
        self.z = np.dot(X[:,:-1],w)  +  b          # z = wx + b
        y_pred = self.sigmoid(self.z)
        djdw = (1/X.shape[0]) * np.dot(X[:,:-1].T,(y_pred - y))     # (1/m) * (f_wb - y)x     .f_wb is the predicted value which is the output of sigmoid
        djdb = (1/X.shape[0]) * np.sum((y_pred - y))

        # adding the regularized part
        djdw += self.regularize(X,self.regularization)
        return djdw, djdb
    

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] - 1)      # matrix of 0s with shape n-1 to remove the last columns for the intercept
        self.bias = 0

        for _ in range(self.iterations):
            djdw, djdb = self.compute_gradients(X, y)
            self.weights = self.weights - self.alpha * djdw
            self.bias = self.bias - self.alpha * djdb

        # Fitting the model with final weights, final bias
        self.z = np.dot(X[:,:-1],self.weights)  +  self.bias          # z = wx + b