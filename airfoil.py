import matplotlib.pyplot as plt
import numpy as np
from numpy import *
# Generate some data
import  pandas  as pd
np.random.seed(0)
df = pd.read_csv('airfoil_self_noise.dat',header=None,encoding='utf-8',delimiter='\t',quoting=3,low_memory=False)
X = df.loc[:,0:4].values#0表示第一行 这里读取数据并不包含表头，要注意哦！
# X = np.random.random((40, 1)) * 2
# print(X)
# y = np.sin(X[:,0]) + X[:,1] ** 2
x11 = X[:,0]
x22 = X[:,1]
x33 = X[:,2]
x44 = X[:,3]
x55 = X[:,4]

y = df.loc[:,5].values
# y_p =  97.69608094448131 + (-x22 + x44 + 292.0396730194589)/(x11*x33*x55 + 10.505320671750765)
y_p = 130 - (167.37)/(x44/(x11*x33*x55) + 5.67)
index = np.argsort(y_p)
xx = np.array(range(len(y)))
plt.scatter(xx,y[index],label = 'real',alpha=0.8,edgecolors='b')
plt.plot(xx,y_p[index],c = 'r',label = 'pred',linewidth=3.5)
plt.title('airfoil_self_noise (Equ 6)')
plt.legend()
plt.savefig('air-6.pdf')
plt.show()