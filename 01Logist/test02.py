# encoding: utf-8

import numpy as np


# 获取数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# 创建Sigmoid函数
def sigmoid(inX):  # inX表示wx
    return 1.0 / (1 + np.exp(-inX))


# 训练参数 获取参数
def gradAscent(dataMatIn, classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose() #1*100
    m, n = np.shape(dataMat)  # 矩阵行列
    alpha = 0.001  # 设置默认步长
    maxCycles = 500  # 循环迭代次数
    weights = np.ones((n, 1))  # n行1列 100*1
    for k in range(maxCycles):
        # 预测值
        h = sigmoid(dataMat * weights) # 100 * 1
        # 真实值-预测值
        error = labelMat - h # 1*100 的 lablelMat  - 100 * 1 的 h ？？？？
        weights = weights + alpha * dataMat.transpose() * error
    return weights


dataMat, labelMat = loadDataSet()
weights = gradAscent(dataMat, labelMat)


# 测试
def result():
    dataMat2 = np.array([1.0, -1.395634, 4.662541])
    prob = sigmoid(dataMat2 * weights)
    print prob
    if prob > 0.5:
        return 1.0
    else:
        return 0


print result()
