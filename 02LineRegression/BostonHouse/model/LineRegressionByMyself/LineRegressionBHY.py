import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# class DealData():

class LinearRegressionBySelf():
    '''
    自己实现的线性回归
    '''
    theta = [] #theta列表

    __include_bias = True #是否包含偏差
    __dimensionless = True #是否标准化

    def __handelExceptionData(self,dataSet):
        '''
        处理异常数据
        去掉全为0的列

        传参：数据集（DataFrame格式）

        返回：新的数据集
        '''
        dataSet = dataSet.replace(0,np.NaN)
        # print(dataSet)
        dataSet = dataSet.dropna(axis=1,how='all')
        # print(dataSet.dropna(axis=1,how='any'))
        # print(dataSet.dropna(axis=0,how='any'))
        dataSet = dataSet.fillna(value=0)
        # print(dataSet)
        return dataSet

    def __dimensionlessAndBias(self,dataSet):
        '''
        根据要求进行 标准化和增加偏项

        返回：新的数据矩阵
        '''
        if self.__dimensionless:
            dataSet = StandardScaler().fit_transform(dataSet)

        if self.__include_bias:
            dataSet = np.column_stack((np.mat(dataSet),np.ones((dataSet.shape[0],1))))
            # print(dataSet)
        
        return dataSet

    def __gradientDestence(self,dataSet,labelSet,alpha = 0.0001,maxCycle=500,precision=1e-6):
        '''
        梯度下降

        传参：
            数据集，标签，最大迭代次数,精确度

        返回：
            theta矩阵
        '''
        dataSet = self.__dimensionlessAndBias(dataSet)
        
        theta = np.ones((dataSet.shape[1],1))
        # dataMat = np.mat(dataSet)

        labelMat = np.mat(labelSet).T

        for i in range(maxCycle):
            h = dataSet * theta
            error = h - labelMat
            theta = theta - alpha * dataSet.T * error

            # if error.sum() < precision:
            #     print(i)
            #     break
        
        self.theta = theta
    
    def __leastSquares(self,dataSet,labelSet,disturbance=0):
        '''
        最小二乘求解

        参数：
            disturbance 扰动大小 （默认为0）
        '''
        labelSet = np.mat(labelSet).T

        dataSet = self.__dimensionlessAndBias(dataSet)
        
        self.theta =np.dot((np.dot(dataSet.T,dataSet) + disturbance * np.identity(dataSet.shape[1])).I , np.dot(dataSet.T,labelSet))
        

        
    def fit(self,dataSet,labelSet,alpha = 0.0001,maxCycle=500,precision=1e-6,dimensionless=True,include_bias=True,disturbance=0,way='GD'):
        '''
        训练数据

        参数： 
            dataSet：数据集
            labelSet：标签
            alpha = 0.0001 学习率
            maxCycle=500 最大循环次数
            precision=1e-6 精确度
            dimensionless=True 标准化处理
            include_bias=True 添加偏项
            disturbance=0 当用最小二乘可以更改扰动大小
            way：训练方式 
                'GD'-梯度下降
                'LS'-最小二乘法
        '''
        self.__include_bias = include_bias
        self.__dimensionless = dimensionless

        if way == 'GD':
            self.__gradientDestence(dataSet,labelSet,alpha=alpha,maxCycle=maxCycle,precision=precision)
            return self.theta
        elif way == 'LS':
            self.__leastSquares(dataSet,labelSet,disturbance)
            return self.theta
    
    def predict(self,dataSet):
        '''
        预测结果
        '''
        dataSet = self.__dimensionlessAndBias(dataSet)

        return np.dot(dataSet,self.theta)
    
    def score(self,dataSet,labelSet):
        '''
        评分
        '''
        return r2_score(self.predict(dataSet),labelSet)