import numpy as np
import matplotlib.pyplot as plt

def loadData(filePath):
    '''
    加载数据
    '''
    with open(filePath) as file:

        all_data = np.mat([line.strip().split('\t') for line in file.readlines()],dtype=float)

    x = all_data[:,:2]
    y = all_data[:,-1]

    # print(x,y.T)
    # print(y.shape)
    return x,y

