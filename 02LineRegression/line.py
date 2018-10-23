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

def paintImg(x,y,*model,**nextDataSet):
    '''
    作图：

    参数： x,y数据集
          model:模式  0 表示直线图  
          
          1表示散点图  
          nextDataSet 下一组数据集 next_X=[] next_Y= [] 默认前直线图，后散点图

          2表示两条线加一个散点图
          nextDataSert 需要传入3组数据 同样 数先直线图，最后散点图
    
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
        plt.annotate('DataSet',xy=(x[3],y[3]),xytext=(x[0],y[3]+0.1),arrowprops=dict(facecolor='blue', shrink=0.01))
        plt.annotate('NormalEquation',xy=(x_new0[11],y_new0[11]),xytext=(x_new0[11],y_new0[11]+0.1),arrowprops=dict(facecolor='red', shrink=0.01))
        plt.annotate('GradientDestence',xy=(x_new1[30],y_new1[30]),xytext=(x_new1[20],y_new1[20]+0.1),arrowprops=dict(facecolor='black', shrink=0.01))

        plt.text(4.75,0,'-:NormalEquation \n ^:GradientDestence')


    
    plt.show()

def normalEquationReularization(x,y):
    '''
    利用 Normal Equation 正归方程求解 Theta (矩阵))

    返回：theta矩阵

    '''
    xTx = np.dot(x.T,x)
    
    if np.linalg.det(xTx) == 0:
        print('Err ! This matrix is singular ,cannot do inverse')
        '''
        对于行列式为0 的矩阵，说明其中有数据特征成比例，对于这种情况，我们可以利用数据压缩的思想进行将维，将
        成比例的特征压缩后进行求解，当然你也可以用pinv的伪逆的方式进行强制执行

        这里代码不给出，只给出说明，在往后的学习中，我们再去探讨,我们只是告诉大家！这类问题，也是有解决方法的。
        '''
        return 
    else:
        return np.dot(xTx.I, np.dot(x.T,y))
        

    
def gradientDestence(x,y):
    '''
    梯度下降算法求解 Theta（矩阵）
    '''
    n = x.shape[1]
    theta = np.ones((n,1))
    alpha = 0.001 
    maxCycle = 1000
    # result = []

    for i in range(maxCycle):

        h =  x * theta

        error = h - y

        theta = theta - alpha * x.T * error
    
        # result.append(theta.tolist())
    
    return theta

x,y = loadData('/Volumes/My_Mac/python_bigdata/bidData/MachineLearning/AI/02LineRegression/ex0.txt')

# print(normalEquationReularization(x,y))
# print(gradientDestence(x,y))
normal_requation_result = normalEquationReularization(x,y)

gradient_destence = gradientDestence(x,y)

h1 = normal_requation_result.T * x.T
h2 = gradient_destence.T * x.T

h1 = h1.flatten().A[0]

h2 = h2[0].flatten().A[0]

x = x[:,1].flatten().A[0]

# paintImg(x,y.flatten().A[0],1)
# paintImg(x,h1,0,1,x_new=x,y_new=y.flatten().A[0])
# paintImg(x,h2,0,1,x_new=x,y_new=y.flatten().A[0])

paintImg(x,h1,0,1,2,x_new0=x,y_new0=h2,x_new1=x,y_new1=y.flatten().A[0])