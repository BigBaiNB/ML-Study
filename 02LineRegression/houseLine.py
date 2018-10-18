import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

path = '/Volumes/My_Mac/python_bigdata/bidData/MachineLearning/AI/02LineRegression/housrPrice.csv'

def loadData(filePath):
    '''
    加载数据
    返回 dataframe数据格式
    '''
    all_dat = pd.read_csv(filePath,sep=',')
    x = all_dat['GrLivArea']
    y = all_dat['SalePrice']

    return x,y

def paintImg(x,y,*model,**nextDataSet):
    '''
    作图：

    参数： x,y数据集
          model:模式  0 表示直线图  
          
          1表示散点图  
          nextDataSet 下一组数据集 next_X=[] next_Y= [] 默认前直线图，后散点图

          2表示其他模式
          nextDataSert 需要传入3组数据 同样 数先直线图，最后散点图

          需要什么模式 就传入什么模式 例 0，1，2
    
    注意：x数据集不能带如偏置 1 且 特征量为 1维
    '''
    
    if len(model) == 1:
        if model[0] == 0:
            plt.plot(x,y)
        
        elif model[0] == 1:
            plt.scatter(x,y)

    elif len(model) == 2:
        figure =  plt.figure().add_subplot(111)
        figure.plot(x,y,'r-')
        x_new,y_new = nextDataSet.values()
        figure.scatter(x_new,y_new)
    
    elif len(model) == 3:
        figure =  plt.figure().add_subplot(111)
        figure.plot(x,y,'r-')
        x_new0,y_new0,x_new1,y_new1 = nextDataSet.values()
        figure.plot(x_new0,y_new0,'k^')
        figure.scatter(x_new1,y_new1)


    
    plt.show()

# x,y = loadData(path)

l = [i for i in range(80,100,4)]
x = random.choices(l,k=10)
x.sort()
l2 = [i for i in range(80,1000,4)]
y = random.choices(l2,k=10)
y.sort()
print(x)
print(y)

paintImg(x,y,1)