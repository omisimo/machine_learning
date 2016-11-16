__author__ = 'eduardomartinez'

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import random as rnd

def entrena(X, y, W, alpha, lam):
    numDatos = len(X)
    for i in range(numDatos):
        sal = salida(W,X[i])
        error = y[i][0] - sal
        W[0] = W[0]+alpha*error
        numAtributos = len(X[i])
        for j in range(numAtributos):
                W[j+1] = W[j+1] + alpha*error*X[i][j] - lam*W[j+1]
    return W
def salida(W, X):
    temp = W[0]
    for i in range(len(W)):
        if i > 0:
            temp = temp + W[i]*X[i-1]
    return temp

def main():
    df = pd.read_csv("/Users/eduardomartinez/PycharmProjects/prueba/regLinPoli.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(df[['X','X2','X3','X4']],df[['y']],train_size=0.75)
    scaleX = preprocessing.StandardScaler()
    scaleX.fit(X_train)
    X = scaleX.transform(X_train)
    scaleY = preprocessing.StandardScaler()
    scaleY.fit(Y_train)
    y = scaleY.transform(Y_train)
    X_t = scaleX.transform(X_test)
    y_t = scaleY.transform(Y_test)

    eta = 0.01
    lam = np.arange(0.001,0.01,0.001)
    error = []
    for l in lam:
        # establecer W's random
        W = [rnd.random(),rnd.random(),rnd.random(),rnd.random(),rnd.random()]
        # ejecutar nuestro entrenamiento n numero de veces
        for i in range(10):
            W = entrena(X,y,W,eta,l)
        #calcular error cuadratico medio
        error_cu = []
        for i in range(len(X_t)):
            y_i = salida(W,X_t[i])
            error_cu.append((y_t[i]-y_i)**2)
        error_real = np.mean(error_cu)
        error.append(error_real)
        print(W,l,error_real)
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot(lam,error)
    plt.show()

if __name__ == "__main__":
    main()