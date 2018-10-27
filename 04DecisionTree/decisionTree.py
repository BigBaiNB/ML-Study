import numpy as np
import math
import matplotlib as plt
import pandas as pd
from pandas import DataFrame as df

def calcAListValueProbability(dataList):
    '''
    说明： 计算一列中的各个元素出现的概率

    传入： 一列数据

    返回：Series 结果
    '''

    #获取信息数量
    result = {}
    for i in dataList:
        result[i] = result.get(i,0)+1
    result = pd.Series(result)

    #获取总信息长度
    data_length = len(dataList)

    #计算概率
    result = result / data_length

    return result

def safe_log(number,base=2,default=0):
    '''
    作用：防止number为0时报错
    
    参数：
        number:待求解数
        base:基底
        default：log0 默认返回0
    返回值：
        计算结果
    '''
    if number==0:
        return default
    
    return math.log(number,base)


def calcEntropy(dataList,coding=2,default=0):
    '''
    说明：计算信息熵
    
    传参:
        dataList:  可迭代数据集
        coding:编码位数 默认 2进制
        default：log0 默认返回0

    返回值：信息熵计算结果

    '''

    result = calcAListValueProbability(dataList)

    #返回结果
    return sum([p*safe_log(1/p,coding) for p in result])
    
# print(calcEntropy(['a','a','b','c']))  

def InformationGain(dataSet,continuous_valued=False,columnName=[],resultColumName=None):
    '''
    说明：计算指定数据集的信息增益

    传参：
        dataSet: 数据集(DataFrame格式)
        continuous_valued:s是否为连续值（默认否）
        columnName: 所需计算信息增益的列名（默认是全部）
        resultColumName: 指定表示结果的列名 (默认最后一列)

    返回：
        各列信息增益的结果
    '''
    result = pd.Series()
    if resultColumName == None:
        resultColumName = dataSet.columns[-1]
    #获取指定列名
    if columnName:
        nameList = columnName
        if resultColumName in columnName:
            print('Warning:你的选择列中包含结果列，已经自动处理掉！') #可以自己抛个异常或warning
            nameList.remove(resultColumName)
    else:
        nameList = dataSet.columns.values.tolist()
        nameList.remove(resultColumName)
    
    baseEntropy = calcEntropy(dataSet[resultColumName])#系统熵

    #计算每列的信息增益
    for oneColumn  in nameList:
        probabilityResult = calcAListValueProbability(dataSet[oneColumn])#获取一列的所有数据的概率
        valueEntropyResult = dataSet.groupby(by=oneColumn)[resultColumName].apply(calcEntropy)#以2为编码计算熵
        # print('valueEntropyResult',valueEntropyResult)
        # print('probabilityResult*valueEntropyResult',probabilityResult*valueEntropyResult)
        result[str(oneColumn)] = baseEntropy - sum(probabilityResult*valueEntropyResult)
    
    return result.sort_values(ascending=False)

# textSet = df([[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']],columns=['No_surfacing','Filppers','Result'])
# print(InformationGain(textSet))

textSet = df([['是', '单身', 1, '否'], ['否', '已婚', 1, '否'], ['否', '单身', 1, '否'], ['是', '已婚', 1, '否'], ['是', '离婚', 0, '否'], ['否', '离婚', 0, '是'], ['否', '单身', 0, '是'], ['否', '已婚', 0, '否'], ['否', '单身', 0, '是'], ['是', '离婚', 1, '否']],columns=['No_surfacing','Filppers','InputMoney','Result'])

print(InformationGain(textSet))

    


