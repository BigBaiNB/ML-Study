# encoding: utf-8

from math import log
import operator


# 创建数据
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


dataSet, labels = createDataSet()


# 计算熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 获取数组的长度
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 获取类别
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 概率
        shannonEnt -= prob * log(prob, 2)  # 熵
    return shannonEnt

#划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的熵和特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 系统熵
    bestInfoGain = 0.0 #初始化
    bestFeature = -1 #信息增益最大的特征（第0个还是第一个）
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 防止重复
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 信息增益
        if (infoGain > bestInfoGain):  # 计算最大信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 字典排序
def majorityCnt(classList): #['yes','yes','no'] => {"yes":2,'no':1}
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#递归拿出决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #取出类别
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则直接返回该类别标签
        return classList[0]
    if len(dataSet[0]) == 1: #?
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  #复制标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

print createTree(dataSet, labels)




