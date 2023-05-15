import numpy as np
import matplotlib.pyplot as plt
# Importing csv module
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
# Generate some data
import  pandas  as pd
# 读取数据
data = pd.read_csv('Skyserver_SQL4_25_2023 6_11_06 AM.csv')
# data = pd.read_csv('GMST_response_1851-2021.csv')
# 计算相关性
print(data)
print('4'*100)
# data = data.values
X = data.iloc[:,[3,4,5,6,7]].values
# print(X[:,1])
# print(X)
y = data.iloc[:,-1].values
#
#### dso1(l = 20,-sin,cos,log) ; dso2(l = 30)
x1 = X[:,0].reshape(-1,1)
x2 = X[:,1].reshape(-1,1)
x3 = X[:,2].reshape(-1,1)
x4 = X[:,3].reshape(-1,1)
x5 = X[:,4].reshape(-1,1)

# y_pred = 0.076 + (66)/(2.17e-24 * np.exp((31.9 * (x3 + 15.58)))/x4 + 60.8)
# y_pred = 1.49 * np.exp((88.30 * x2 * x3 * (11 - x1))/(x1 + np.exp(0.67 * x4)))
y_pred = (11.77*np.log(np.log(40.69 * np.exp(-(38.52 * x3)/(x4) + 2 * x4) + 3.05)))/(x3)
# y_pred = (0.01 * x2)/(551003 * x1 * np.exp(x3 - 1.94 * x4) + (x4 - 17.40)/(x2))
y_pred = np.array(y_pred.reshape(1,-1))[0]
xx = np.array(range(len(y)))
index = np.argsort(y_pred)
# plt.scatter(xx,y[index],label = 'real',alpha=0.12,edgecolors='b')
# plt.plot(xx,y_pred[index],c = 'y',label = 'pred',linewidth=2)

plt.plot(xx,y_pred[index],c = 'y',label = 'pred',linewidth=2,alpha=0.7)
plt.scatter(xx,y[index],label = 'real',alpha=0.2)
plt.show()
