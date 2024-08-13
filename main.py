import numpy as np
import pandas as pd
from data_loader import load_file
from preprocessing import preprocessor
from regularization import LogisticGradientRegularization
from logistic_regression import LogisticRegression
from gradient_descent import GradientDescent
from evaluation import Evaluator

def main():

    # load data as a dataframe
    dataobj = load_file('diabetes_data.csv')
    data = dataobj.load()

    # preprocessing it
    processdata_obj = preprocessor(data)
    X_train, X_test, y_train, y_test = processdata_obj.preprocess()

    
    print('\nUSING LOGISTIC REGRESSION')
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train,y_train)
    print('Evaluating training data: ')
    y_pred = logreg_model.predict(X_train)
    evaluator = Evaluator()
    evaluator.evaluate(y_train, y_pred)
    print('Evaluating Testing data')
    y_pred = logreg_model.predict(X_test)
    evaluator = Evaluator()
    evaluator.evaluate(y_test, y_pred)


    print('\n\nUSING GRADIENT DESCENT')
    gd_model = GradientDescent(0.01,1000)
    gd_model.fit(X_train,y_train)
    print('Evaluating training data: ')
    y_pred = gd_model.predict(X_train)
    evaluator = Evaluator()
    evaluator.evaluate(y_train, y_pred)
    print('Evaluating Testing data')
    y_pred = gd_model.predict(X_test)
    evaluator = Evaluator()
    evaluator.evaluate(y_test, y_pred)


    print('\n\nUSING REGULARIZATION AND GRADIENT DESCENT OF LOGISTIC REGRESSION')
    reg_model = LogisticGradientRegularization(lr = 0.001, 
                  iterations = 1000, 
                  regularization = 'l2', 
                  lambda_ = 10)
    reg_model.fit(X_train,y_train)
    print('Evaluating training data: ')
    y_pred = reg_model.predict(X_train)
    evaluator = Evaluator()
    evaluator.evaluate(y_train, y_pred)
    print('Evaluating Testing data')
    y_pred = reg_model.predict(X_test)
    evaluator = Evaluator()
    evaluator.evaluate(y_test, y_pred)

if __name__ == '__main__':
    main()