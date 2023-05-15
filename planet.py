import matplotlib.pyplot as plt
import numpy as np
from numpy import *
# Generate some data
import  pandas  as pd
import xml.etree.ElementTree as ET, urllib.request, gzip, io
import csv
data = pd.read_csv('planet.csv')
# data = pd.read_csv('GMST_response_1851-2021.csv')
# 计算相关性
# print(data)

X = data.iloc[:,[1,2,3]].values
# print(X[:,1])
# print(X)
y = data.iloc[:,0].values

x1 = X[:,0].reshape(-1,1)
x2 = X[:,1].reshape(-1,1)
x3 = X[:,2].reshape(-1,1)

# print(x1)
# print(x3)
# print(y)
ind = y<=5
y = y[ind]
y = np.array(y)
# y_pred = x1 * (1.21 + (2.52e-5)/(cos(x1 + x1/(x2+6.15) + 2) + 1))
# y_pred = x1 + 0.12  + (0.027)/(2 * x1 - 2 * np.exp(x1) + 8.24 - (4.25)/(x1)) - 0.35
# y_pred = x1 + 0.132 - (0.00426)/(-1.1125*x1 + (1.132 *(x2 - 1.01))/(x1))
# y_pred = -x1*(x1 - 2.6 + (0.002)/(x1*(x2 - 0.726*np.exp(x1)))) - 0.214
y_pred = 1.1*x1 + (0.015)/(-x1**2 + x1 + (0.7)/(x1))
y_pred = np.array(y_pred.reshape(1,-1))[0]
y_pred = y_pred[ind]

ind2 = y_pred<=5
y = y[ind2]
y_pred = y_pred[ind2]

xx = np.array(range(len(y)))
index = np.argsort(y_pred)

# sort
plt.plot(xx,y_pred[index],c = 'y',label = 'pred',linewidth=2,alpha=0.7)
plt.scatter(xx,y[index],label = 'real',alpha=0.9)
# no sort
# plt.plot(xx,y_pred,c = 'y',label = 'pred',linewidth=2,alpha=0.7)
# plt.scatter(xx,y,label = 'real',alpha=0.9)
plt.show()