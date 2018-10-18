import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import random

def sigmod_func(x):
    '''
    sigmod函数

    传参：x
    '''
    return 1 / (1 + np.exp(-x))

# x = np.linspace(-10,10,10000)
# y = [sigmod_func(i) for i in x]

def print_img(x,y,title):
    '''
    画图
    '''
    plt.title(title)
    plt.plot(x,y,'r-')
    plt.axis()
    # plt.plot(np.linspace(-10,10,100),[1 for i in range(100)],'k:')
    plt.ylabel = 'y'
    plt.xlabel = 'x'

    ################ 移动坐标位置 ###########

    # ax = plt.gca()
    # #去边框
    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')
    # #移动位置到原点
    # ax.xaxis.set_ticks_position('bottom')
    # ax.spines['bottom'].set_position(('data',0))

    # ax.yaxis.set_ticks_position('left')
    # ax.spines['left'].set_position(('data',0))
    plt.show()

#获取数据
def loadDataSet():
    '''
    获取数据
    '''
    dataMat = []
    labelMat =[]

    fr = open('/Volumes/My_Mac/python_bigdata/MrWu/marchin_study/logicst/testSet.txt')

    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
        labelMat.append([int(lineArr[-1])])
    return dataMat,labelMat
'''
验证cost是否为线性
'''
def check_cost_line():
    x,y=loadDataSet()
    # fig = plt.figure()
    # result_3D = fig.gca(projection='3d')
    # x = 2
    # print(y)
    n = len(x)
    theta_1 = np.linspace(-0.5,-0.2,n//10)
    # theta_2 = np.linspace(-0.3,-0.1,n)
    x = np.mat(x)
    y = np.mat(y)
    # print(x.shape)
    cost = []
    for i in theta_1:
        for j in theta_1:
        # print(temp_theta)
        # cost.append((sigmod_func(temp_theta*x)[0,0]-random.choice([0,1]))**2)
            cost.append(np.sum([i[0,0]**2 for i in (sigmod_func(x*(np.mat([[1],[i],[j]])))-y)]))
        # print((sigmod_func(x*i)).shape)
        # break
    # print(cost)
    # print_img(theta,y,'cost')
    # print(len(y))
    # print(theta_1.shape)
    # print(np.array(y).shape)

    # result_3D.get_xlabel = 'theta'
    # result_3D.plot(theta_1,y,np.linspace(0,1,100))

    # result_3D.plot()

    # print(x.shape)
    # print(y.shape)
    # print(len(cost))

    plt.plot(np.linspace(-1,0,n),cost)
    plt.xlabel='theta'
    plt.ylabel='cost'
    plt.show()
check_cost_line()
# print(J_result)
        
