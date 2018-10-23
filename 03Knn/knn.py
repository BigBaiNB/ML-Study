import numpy as np
import operator

#数据集
def loadData():
    originalData = np.array([[1,0],[1,1],[0,0],[0,1]])
    classResult = ['A','A','B','B']
    return originalData,classResult

originalData,classResult = loadData()

#KNN
def knnAlgorithm(originalData,classResult,predictedData,k):
    '''
    说明：KNN核心算法（理解用）
    
    传参：
        originalData:原始数据集
        classResult：原始数据集的分类结果
        predictedData:需要预测的数据
        k:范围
    
    返回值：
        分类结果
    
    注意：此代码仅仅方便理解！
    
    '''
    #第一步：获取原始数据集每个点与待预测数据的距离(为了减少计算，就没开根号，并不影响结果！)
    distance_list = {key:np.square(predictedData[0]-i[0]+np.square(predictedData[1]-i[1])) for key,i in enumerate(originalData)}
    #根据距离排序，取出前K项
    sort_distance = sorted(distance_list.items(),key=operator.itemgetter(1))[:k]
    #获取最多的一类，并返回！
    result = {}
    for key,distance in sort_distance:
        result[key] = result.get(key, 0) + 1

    return classResult[sorted(result.items(),key=operator.itemgetter(1),reverse=True)[0][0]]
    
print(knnAlgorithm(originalData,classResult,[1,1],3))

def knnAlgorithm2(originalData,classResult,predictedData,k):
    '''
    说明：KNN核心算法（矩阵方式）
    
    传参：
        originalData:原始数据集
        classResult：原始数据集的分类结果
        predictedData:需要预测的数据
        k:范围
    
    返回值：
        分类结果
    
    '''
    #获取行数
    data_line_number = originalData.shape[0]

    #扩展(注意，默认列扩展，所以需要（x，1）的方式)
    extend_predicted_data = np.tile(predictedData,(data_line_number,1))

    #求差累和
    distance_all = (np.square(originalData - extend_predicted_data)).sum(axis=1) ** 0.5
    

    #返回前K项
    sort_distance = distance_all.argsort()[:k]

    #获取最多的一类，并返回！
    result = {}
    for key in sort_distance:
        result[key] = result.get(key, 0) + 1

    return classResult[sorted(result.items(),key=operator.itemgetter(1),reverse=True)[0][0]]


print(knnAlgorithm2(originalData,classResult,[1,1],3))



