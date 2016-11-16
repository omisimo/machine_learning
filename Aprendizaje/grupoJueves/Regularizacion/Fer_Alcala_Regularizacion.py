# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:24:51 2015

@author: fernanda
"""
#cd /Users/fernanda/Dropbox/batmelon/aprendizaje/Aprendizaje/datos


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df3=pd.read_csv("regLinPoli.csv")


X_train, X_test, Y_train, Y_test = train_test_split(df3[['X','X2','X3','X4']],df3['y'],train_size=0.85)

scaleX=StandardScaler()
scaleY=StandardScaler()
scaleX.fit(X_train)
X_train=scaleX.transform(X_train)
scaleY.fit(Y_train)
Y_train=scaleY.transform(Y_train)


X_train = np.array(X_train);Y_train = np.array(Y_train)
X_test = np.array(X_test);Y_test = np.array(Y_test)

w=np.array([10.0,10.0,10.0,10.0,10.0])

def salida(w,x):
    w = np.array(w)
    x = np.array(x)
    x=np.insert(x,0,1)
    res = float(np.dot(x,w))
    return(res)
    
def entrena(w,X_train,Y_train,eta=0.000001):
    errores=[]
    for i in range(len(X_train)):
        errores.append(Y_train[i] - (salida(w,X_train[i]) ))
        w_anterior=np.array([element for element in w])
        new_X_train=np.insert(X_train,0,1)
        for i in range(len(w)):   
            w[i] = w[i] + eta * ( Y_train[i] - (salida(w_anterior,X_train[i]) ))*new_X_train[i]
    return(errores,w)
    
errores,w=entrena(w,X_train,Y_train,eta=0.0000001)    

scaleX=StandardScaler()
scaleY=StandardScaler()
scaleX.fit(X_train)
X_test=scaleX.transform(X_test)
scaleY.fit(Y_train)
Y_test=scaleY.transform(Y_test)

def errores_test(x,y,w):
    errores_test=[]
    for i in range(len(x)):
        errores_test.append(y[i] - (salida(w,x[i]) ))
    return errores_test

errores_test=errores_test(X_test,Y_test,w) 

plt.plot(range(len(X_train)),errores,color='blue')

plt.plot(range(len(X_test)),errores_test,color='green')