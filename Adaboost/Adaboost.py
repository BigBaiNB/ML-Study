import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadData():
    '''
    加载数据
    '''
    datMat = np.matrix([[1.,  2.1],
                    [2.,  1.1],
                    [1.3,  1.],
                    [1.,  1.],
                    [2.,  1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] >threshVal] = -1.0
    return retArray


def bulidStump(dataArr,classLables,D):
    '''
    创建最佳决策树（分类器）
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLables).T
    m,n = np.shape(dataMatrix)
    numSteps = m #总步数
    bestStump = {}#最优决策树
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf #无穷大

    for i in range(n):
        rangeMin = dataMatrix[:,i].min() #特征最小的值
        rangeMax = dataMatrix[:,i].max() #特征值最大的
        stepSize = (rangeMax - rangeMin)/numSteps #步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                predicteVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predicteVals == labelMat] = 0
                weightedError = D.T * errArr

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predicteVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLables,numlt=40):
    '''
    数据集 类别标签 迭代次数
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]

    D = np.mat(np.ones((m,1))/m)#初始化
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numlt):
        #获取分类器 错误率 预测类别
        bestStump,error,ClassEst = bulidStump(dataArr,classLabels,D)#选择最好的分类

        #计算话语权
        alpha = float(0.5*np.log((1-error)/max([error,1e-16])))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        expon = np.multiply(-1*alpha*np.mat(classLabels).T,ClassEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()

        aggClassEst += alpha*ClassEst

        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        # print(aggClassEst)
        # print(np.sign(aggClassEst))
        # break
        errorRate = aggErrors.sum()/m

        if errorRate == 0.0 : break
    return weakClassArr,aggClassEst

dataM,classLabels = loadData()
print(adaBoostTrainDS(dataM,classLabels))

