#!/usr/bin/env python
#-*- coding:utf8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import re
import jieba
import os

dataPath = '/Users/apple/AiTrain/bayes/news.csv'
jiebaDictionaryPath = r'‎⁨/Volumes/My_Mac/stopWords/dict.txt.big'
stopWordsPath = r'/Volumes/My_Mac/stopWords/stopwordsAll.txt'
current_dir = os.path.abspath(jiebaDictionaryPath)
dict_file = os.path.join(current_dir,'dict.txt.big')
# jieba.set_dictionary(dict_file)
print(current_dir)

class getInformation():

    def __init__(self,dataPath=dataPath,jiebaDictionaryPath=jiebaDictionaryPath,stopWordsPath=stopWordsPath):
        
        self.stopWords = self.__setStopWord(stopWordsPath)
        self.Data = self.loadData(dataPath)
    
    @staticmethod
    def setJiebaDictionary(jiebaDictionaryPath):
        '''
        更改jieba分词字典

        参数：
            结巴分词字典
        '''
        jieba.set_dictionary(jiebaDictionaryPath)

    def __setStopWord(self,stopWordsFilePath):
        '''
        设置停顿词

        参数：
            stopWordsFilePath：停顿词路径

        返回值：
            停顿词 set()
        '''
        stopWordsSet = set()
        
        with open(stopWordsFilePath) as f:
            for word in f.readlines():
                stopWordsSet.add(word.strip())

        # self.setStopWord = stopWordsSet

        return stopWordsSet

    def changeStopWords(self,stopWordsFilePath):
        '''
        修改停顿词

        参数：
            stopWordsFilePath：停顿词路径

        返回值：
            无
        '''
        self.stopWords = self.__setStopWord(stopWordsPath)


    def loadData(self,filePath,index_col=0,header=0):
        '''
        读取数据
        '''
        return pd.read_csv(filePath,engine='python',index_col=index_col,header=0)

    def getClassify(classList):
        '''
        说明：自动获取分类表
        
        传参：可迭代对象

        返回值： series 数据
        '''
        return pd.Series(classList,name='ClassifyResult').unique()

    def splitDoucumentToWords(doucumnts,stop_words=[]):
        '''
        说明：分词

        传参：
            doucuments：文档（每一行为一篇文档）
            stop_words：不需要的词 (默认为空)
        '''
        regRule = re.compile(r'')

        
        for doucumnt in doucumnts:
            jieba.cut(doucumnt)#默认精确模式，并利用隐马尔可夫
            

    
getInformation()