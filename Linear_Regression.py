# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 20:42:38 2022

@author: linhg
"""
#include library
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame,Series
from sklearn.linear_model import LinearRegression
#Load data from file Exel using Panda Library
Cars_Path = r"C:\Users\linhg\OneDrive\Máy tính\HW AI\HW4\Salary_Data.csv"
df = pd.read_csv (Cars_Path)
print (df)


X = df.iloc[:, 0].values.reshape(-1, 1) # values converts it into a numpy array
Y = df.iloc[:, 1].values.reshape(-1, 1) # -1 means that calculate the dimension of rows, but have 1 column
# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y)
w = np.dot(np.linalg.pinv(A), b)
print ('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 30, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X.T, Y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([0, 11, 30000, 130000])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

#Dung sklearn de tim nghiem
from sklearn import datasets, linear_model
# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, Y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)

#Drawing the fitting line
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
