#!/usr/bin/env python
#-*- coding:utf8 -*-
import pandas as pd
import numpy as np
from pandas import DataFrame as df

def loadData():
    '''
    装载数据
    '''
    dataSet = [
            ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
        ]
    classVec = [0,1,0,1,0,1] # 1 代表侮辱性文字 ，0代表正常言论
    return dataSet,classVec

def createVocabList(dataSet):
    '''
    获取单词本
    '''
    vocabSet = set()

    for document in dataSet:
        vocabSet = vocabSet | set(document)
    
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    '''
    构建单词向量 词集模型
    '''
    returnVec = [0] *len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    #转化为list 避免无序
    return list(returnVec) 


def trainNBO(trainMatrix,trainCategory):
    '''
    返回：
        侮辱性文本中各个单词概率 P（W | c=1）
        非侮辱性文本各个单词概率 p(W | c=0)
        侮辱性文本概率 P（C = 1）
        非侮辱性概率 P(C=0) = 1 - P(C=1)
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs) #计算侮辱性概率
    p0Num = np.ones(numWords);p1Num = np.ones(numWords) #分子
    p0Denom = 2 ; p1Denom = 2 #分母

    for i in range(numTrainDocs):#遍历
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] #统计侮辱性的单行文本 单词量 保持数组形式
            p1Denom += sum(trainMatrix[i]) #总词汇
        else:
            p0Num += trainMatrix[i]
            # print('trainMatrix',trainMatrix[i])
            # print('p0Num',p0Num)
            p0Denom += sum(trainMatrix[i])
            # print('p0Denom',p0Denom)
    p1Vect = np.log(p1Num/p1Denom) #为了避免不好算 毕竟除法精度不高
    p0Vect = np.log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''
    贝叶斯判断
    '''
    p1 = sum(vec2Classify * p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec)+np.log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadData()#装载数据
    myVocabList = createVocabList(listOPosts)#加载单词表
    
    trainMat = [] #获取单词向量
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    
    p0V,p1V,pAb = trainNBO(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))

    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()

def bagOfW2V(vocabList,inputSet):
    '''
    词袋模型 w2v
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    
    return returnVec


