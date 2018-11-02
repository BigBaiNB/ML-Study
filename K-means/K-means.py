import numpy as np
import pandas as pd
X = np.array([[1, 2], [2, 2], [6, 8],[7 ,8],[70,80]])
C = np.array([[1.0, 2.0], [2.0, 2.0],[72,82]]) #聚类中心 
classType = len(C) #类别数
iters = 5 #迭代次数

while iters>0:
    distance = []
    for c in C :#遍历每一个聚类中心，计算样本到每个聚类中心的距离
        a = np.sum((X - c) **2,axis=1)
        distance.append(a)
    distance = pd.DataFrame(data = distance)
    classification = distance.apply(pd.Series.idxmin,axis=0)#确定分类
    
    for i in range(classType):#获取每个簇的样本，求质心更新C
        point = X[classification==i]
        C[i] = np.mean(point,axis=0)
    iters -=1
print(C)
print(classification)

''' 
小思考:
    只给定样本个数 ，按个数多少为概率进行随机抽样 
'''
import random
import pandas as pd
a = [["A",10],["B",20],["C",30],["D",40]]
#第二个元素越大，采到的概率越大，请采样1w次，然后统计频率
res = []
for i in range(10000):
    s = sum([i[1] for i in a]) * random.random()
#     random.shuffle(a)
    sums = 0
    for i in a:
        sums += i[1]
        if(sums>s):
            res.append(i[0])
            break
import pandas as pd
pd.Series(res).value_counts()

'''
调包
'''
import numpy as np
X = np.array([[1, 2], [2, 2], [6, 8],[7 ,8]])
C = np.array([[1, 1], [2, 1]])
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
model.fit(X,C)
model.cluster_centers_
labels = model.labels_
from sklearn.metrics import silhouette_score #轮廓系数法
silhouette_score(X, labels, metric='euclidean')
# model.inertia_
# model.inertia_