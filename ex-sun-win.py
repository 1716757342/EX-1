import matplotlib.pyplot as plt
import numpy as np
from numpy import *
# Generate some data
import  pandas  as pd
np.random.seed(0)

df = pd.read_excel('Solar station site 5 (Nominal capacity-110MW).xlsx',header=None, dtype=float)
print(shape(df))
X = df.loc[:,0:4].values #0表示第一行 这里读取数据并不包含表头，要注意哦！
# X = np.random.random((40, 1)) * 2
print(shape(X))
#### dso7(l = 20,s - sin,cos)
x1 = X[:,0].reshape(-1,1)
x2 = X[:,1].reshape(-1,1)
x3 = X[:,2].reshape(-1,1)
x4 = X[:,3].reshape(-1,1)
x5 = X[:,4].reshape(-1,1)

y_pre = x1 * (0.000436 * x5*(-3e-5*x1*x4 + 0.31) + 0.093) - 0.2466
y_pre = np.array(y_pre.reshape(1,-1))[0]
y = (df.loc[:,5].values)
print(len(y))
print(y_pre)
# print(y_pre)
print(y)
index = np.argsort(y_pre)
xx = np.array(range(len(y)))

plt.scatter(xx,y[index],label = 'real',alpha=0.12,edgecolors='b')
plt.plot(xx,y_pre[index],c = 'y',label = 'pred',linewidth=3.4)
# plt.scatter(xx,y,label = 'real',alpha=0.12)
# plt.plot(xx,y_pre,c = 'y',label = 'pred',linewidth=2)
# plt.scatter(xx,y,c = 'y')

plt.legend()
plt.title("Solar power generation forecast")
plt.savefig("sun_e.pdf")
plt.show()



# df = pd.read_excel('Wind farm site 1 (Nominal capacity-99MW).xlsx',header=None, dtype=float)
# print(shape(df))
# X = df.loc[:,6:8].values #0表示第一行 这里读取数据并不包含表头，要注意哦！
# # X = np.random.random((40, 1)) * 2
# print(shape(X))
#
# ####dso8(l=20)dso9(l = 20,S-sin,cos
# ####dso8(l = 12,S-sin,cos)
# x1 = X[:,0].reshape(-1,1)
# x2 = X[:,1].reshape(-1,1)
# x3 = X[:,2].reshape(-1,1)
# y_pre = x1*(0.0123 * x1 + (0.0123*(x1-1.53))/(0.00049 * x1 * (x1 - 15.93)+0.049))
# # y_pre = 0.29 * (25.89 - 0.397 * x2) * np.exp(-1.07 * np.sin(0.47*x1)) - 33.0 * cos(0.189*x1) + 27.11
# y_pre = np.array(y_pre.reshape(1, -1))[0]
# y = (df.loc[:,11].values)
#
#
# index = np.argsort(y_pre)
# xx = np.array(range(len(y)))
#
# plt.scatter(xx,y[index],label = 'real',alpha=0.05,edgecolors='b',)
# plt.plot(xx,y_pre[index],c = 'y',label = 'pred',linewidth=3.4)
# # plt.scatter(xx,y,label = 'real',alpha=0.12)
# # plt.plot(xx,y_pre,c = 'y',label = 'pred',linewidth=2)
#
# plt.title("Wind power generation forecast")
# plt.legend()
# plt.savefig("win_e.pdf")
# plt.show()
