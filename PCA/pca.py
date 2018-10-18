from sklearn.decomposition import PCA

'''
自己实现 PCA
'''
import numpy as np

x=np.array([[10001,2,55], [16020,4,11,], [12008,6,33], [13131,8,22]])
pca = PCA(n_components=2)
pca.fit(x)
pca.transform(x)

from sklearn.preprocessing import StandardScaler
#作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。 
X_scaler = StandardScaler()
x_new = X_scaler.fit_transform(x)
pca.fit(x_new)
pca.transform(x_new)

#cov matrix 协方差矩阵
m = len(x_new) # 获取行数
m = np.shape(x_new)[0] # 获取行数 两种方式均可
m = x_new.shape[0]
cov_mat = np.dot(x_new.transpose(),x_new)/(m-1) # 协方差矩阵

#cov matrix 协方差矩阵 方法二
cov_mat = np.cov(x_new)

# svd
sigma = cov_mat
[U,S,V] = np.linalg.svd(sigma) # 奇异值分解
Ur = U[:,0:2] #降维
z = np.dot(x, Ur)

np.dot(z, Ur.transpose())# 数据复原，即x
(S[0]+S[1])/(S[0]+S[1]+S[2])# 从三维降到二维，保留了99.997%的差异

'''
PCA 包
'''
x=np.array([[10001,2,55], [16020,4,11,], [12008,6,33], [13131,8,22]])
X_scaler = StandardScaler()
x_new = X_scaler.fit_transform(x)
pca = PCA(n_components=0.9)# 保证降维后的数据保持90%的信息
pca.fit(x_new)
pca.transform(x_new)

