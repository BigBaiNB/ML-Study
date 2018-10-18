import numpy as np

#获取数据
def loadDataSet():
    dataMat = []
    labelMat =[]

    fr = open('/Volumes/My_Mac/python_bigdata/MrWu/marchin_study/logicst/testSet.txt')

    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
        labelMat.append([int(lineArr[2])])
    return dataMat,labelMat

#sigmod 函数
def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

# 训练参数
def gradAscent(dataMatIn,classLables):
    dataMat = np.mat(dataMatIn)
    lableMat = np.mat(classLables)

    # print(f'labelMat:{labelMat}')

    # n = np.shape(dataMat)[1] #只获取列数
    m,n = np.shape(dataMat)#同时获取 行数 列数

    alpha = 0.001 #设置我们的学习速率和步长

    maxCycle = 500 #设置我们的迭代次数

    weights = np.ones((n,1)) #初始化我们的weight参数

    for k in range(maxCycle):

        h = sigmoid(dataMat * weights) #预测值
        # print(f'第{k}步的预测值为{h}')

        error = lableMat - h #极大似然
        #error = h - lableMat # 梯度求解

        # print(f'error的值为：{error},error的shape{np.shape(error)}')

        weights = weights + alpha * dataMat.transpose() * error
        # weights = weights - alpha * dataMat.transpose() * error * 1 / (2 * m)
    
    return weights

dataMat, labelMat = loadDataSet()
weights = gradAscent(dataMat, labelMat)
print(f'w = {weights}')

'''
w = [[ 4.12414349]
 [ 0.48007329]
 [-0.6168482 ]]
'''