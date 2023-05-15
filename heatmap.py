import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xlrd
# 创建一个包含 8 个特征的数据集
# df = pd.read_excel('Concrete_Data 2.xls')
# df = pd.read_csv('yacht_hydrodynamics.data',header=None,sep='') #### mass radius period temperature
# columns = ['feature_{}'.format(i) for i in range(8)]
# df = pd.DataFrame(data=data, columns=columns)
# df = pd.read_excel('Wind farm site 1 (Nominal capacity-99MW).xlsx',header=None, dtype=float)
df = pd.read_csv('airfoil_self_noise.dat',header=None,encoding='utf-8',delimiter='\t',quoting=3,low_memory=False)
# df = df.values
# df = df[[6,7,8,11]]
print(df)
# 计算相关系数矩阵
corr_matrix = df.corr()
cols = ['x_1','x_2','x_3','x_4','x_5','y']
# 使用热力图可视化相关系数矩阵
sns.heatmap(corr_matrix, annot=True ,cmap='Blues_r', xticklabels=cols, yticklabels=cols)
plt.tight_layout()
plt.savefig('air-h.pdf')
plt.show()
