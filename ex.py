# Import standard packages
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Set time seed
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
# Generate some data
import  pandas  as pd
np.random.seed(0)
# df = pd.read_excel('Win',header=None, dtype=float)
df = pd.read_excel('Concrete_Data.xls',header=None)
print(shape(df))
# X = df.loc[:,0:7].values #0表示第一行 这里读取数据并不包含表头，要注意哦！
X = df.loc[:,[0,1,3,4,7]].values #0表示第一行 这里读取数据并不包含表头，要注意哦！
# X = np.random.random((40, 1)) * 2
print(shape(X))
print('xxxxxx')
####dso6(l = 25)
# X = np.random.random((40, 1)) * 2
print(shape(X))
print('xxxxxx')
####dso8(l=20)dso9(l = 20,S-sin,cos
####dso8(l = 12,S-sin,cos)
x1 = X[:,0].reshape(-1,1)
x2 = X[:,1].reshape(-1,1)
x3 = X[:,2].reshape(-1,1)
x4 = X[:,3].reshape(-1,1)
x5 = X[:,4].reshape(-1,1)
# x6 = X[:,5].reshape(-1,1)
# x7 = X[:,6].reshape(-1,1)
# x8 = X[:,7].reshape(-1,1)

# print(len(x6))
y = (df.loc[:,8].values)

# y_pred = -0.5 + ((2.86 * log(x1 * (x8 - 0.57)) - 10.57)*(x1 + x2 + 0.86 * x3 + 107))/x4
# y_pred = (-0.29 * (x1 + x2 + x3 + 63.7))/(x4*(-0.014 - (22.37)/(x8 * (x1 - 66.8))))
# y_pred = (19.54*x1 + 3.65*x2 + 12.33*x3 + 3.65*(x2 + 359.64) * log(x8) - 5024.94)/x4
# y_pred = 0.058 * x1 + 0.058 * x2 - 0.2 * x3 + 0.058*(x4 + 142)*log(x1*x5) - 23.6
# y_pred = 0.1 * x1 + 0.065 * x2 + x4 - 41.4 + (36.8 * (44.55 - 0.04*x4**2)*(np.log(x5)+1.4))/(x3)
y_pred = 0.66 * x4 + (0.66*(x1 + x2 -x3 + 334))/(7.17 + (17746.56)/(x1*x5))
y_pred = np.array(y_pred.reshape(1,-1))[0]
xx = np.array(range(len(y)))
index = np.argsort(y_pred)

# Plot figure
ax = sns.regplot(x=xx, y=y_pred[index], ci=0.1)
plt.show()

