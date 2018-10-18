import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from LineRegressionByMyself import LineRegressionBHY as BhyLine #我自己的包

from sklearn.model_selection import train_test_split

data_file_path = r'/Volumes/My_Mac/python_bigdata/bidData/MachineLearning/AI/02LineRegression/BostonHouse/data/boston_housing.data'

def loadData(file_path):
    '''
    装载数据

    返回：数据集和数据标签
    '''
    all_date = pd.read_csv(file_path,delim_whitespace=True,header=None,index_col=None)
    return all_date.iloc[:,:-1] , all_date.iloc[:,-1]

data_set,label_set = loadData(data_file_path)

train_data,test_data,train_label,test_label = train_test_split(data_set, label_set, test_size=0.2, random_state=0)

# 我的线性回归
print('我的线性模型')
s = BhyLine.LinearRegressionBySelf()
s.fit(train_data,train_label,way='LS')
print('最小二乘求解后的准确率：',s.score(test_data,test_label))
# print('theta:',s.theta)

s.fit(train_data,train_label,way='DG')
print('梯度下降后的准确率：',s.score(test_data,test_label))
# print('theta',s.theta)

print('****************')
#包
print('包的预测')
lr = LinearRegression()
lr.fit(train_data,train_label)
# lr.save()
print('包的准确率：',lr.score(test_data,test_label))
# print(lr.coef_,lr.intercept_)

'''
模型保存与加载
'''
# from sklearn.externals import joblib
# pppp = '/Volumes/My_Mac/python_bigdata/bidData/MachineLearning/AI/02LineRegression/BostonHouse/model/s.model'
# joblib.dump(s,pppp)
# s_new = joblib.load(pppp)
# print(s_new.score(test_data,test_label))

