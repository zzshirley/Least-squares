#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:04:07 2018

@author: xiaotong
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            current = [item for item in line.split()]
            data.append(current)
    return data
data=load_data('ex1data.txt')
data=np.array(data,np.float)

def featureNormalize(X):
    X_norm = X;
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    for i in range(X.shape[1]):
        mu[0,i] = np.mean(X[:,i]) # 均值
        sigma[0,i] = np.std(X[:,i])     # 标准差
    X_norm  = (X - mu) / sigma
    return X_norm
def standRegress(x,y):  
    xMat=np.mat(x)  
    yMat=np.mat(y).T  
    xTx=xMat.T*xMat  
    if np.linalg.det(xTx)==0.0:  
          
        return  
    ws=xTx.I*(xMat.T*yMat)  
    return ws 
x = data[:,(0,1)].reshape((-1,2))
y = data[:,2].reshape((-1,1))
plt.plot(x,y,'o')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s,Area of City in Square kilometre')
x1=x[:,0]
x2=x[:,1]
fig = plt.figure()
x= np.hstack([np.ones((x.shape[0], 1)),x])   
x=x[:,(0,1,2)]
y=y[:,0]
#print(X) 
w=standRegress(x,y)
x1=x[:,1]
x2=x[:,2]
a=np.array(w[0])
b=np.array(w[1])
c=np.array(w[2])
a=a[0]
y0=a[0]+b[0]*x1+c[0]*x2
cor=np.corrcoef(y,y0)
print(cor)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot3D(x1,x2,y0)
ax.scatter(x1, x2, y, color='red')  
ax.plot(x1, x2, y0, color='black')  
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population,Area')
yt=y0=a[0]+b[0]*6.1101+c[0]*62.69120441
print(w)
