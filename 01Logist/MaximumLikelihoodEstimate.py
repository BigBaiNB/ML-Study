import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
'''
极大似然 MLE

举个小例子：
假如一个罐子里有黑白两种颜色的球，数目和比例都不知道。
假设进行一百次有放回地随机采样，每次取一个球，有七十次是白球。
问题是要求得罐中白球和黑球的比例？
我的选择黑球的概率为 P 白球为 1-p
'''


def load_data():
    '''
    作用：装载数据

    说明：这数据随机给的，我们将文件中最后一列100个数据 0，1 做为我们的黑和白
    '''
    data_path = r'/Volumes/My_Mac/python_bigdata/MrWu/marchin_study/logicst/testSet.txt'
    original_data = open(data_path)

    # 获取我们的 100次黑白球的结果，存入result_data中
    result_data = []
    for line in original_data.readlines():
        lineArr = line.strip().split()
        result_data.append(int(lineArr[-1]))

    return result_data


def distribution_function(p, y):
    '''
    我们的分布函数 y_i 表示第i个球的黑白情况
    '''
    return (p**y) * ((1-p)**(1-y))


def judge_MLE_convex_or_concave(y):
    '''
    判断我们的极大似然函数的凹凸性

    传入：
        y：我们的结果矩阵
    '''
    p_list = np.linspace(0, 1, 1000)  # 生成一个0-1的100个关于p的概率 的列表

    result = []  # 存储我们计算出的估计的概率值

    for p_i in p_list:
        '''
        用不同的p 去计算我们的似然函数的结果
        '''
        result_out = 1  # 输出值

        for y_j in y:
            result_out *= distribution_function(p_i, y_j)  # 累乘运算，就是我们的似然函数

        result.append(result_out)  # 保存该值

    # 熟悉python的同学可以这么写
    # result = [distribution_function(p_i,y_j) for p_i in p_list for y_j in y]

    plt.plot(p_list, result)
    plt.ylabel = 'result'
    plt.xlabel = 'p'
    plt.show()


'''
现在我们来看看这似然函数的凹凸性
'''
# judge_MLE_convex_or_concave(load_data())
