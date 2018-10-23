# encoding: utf-8
import numpy as np
from scipy.stats import multivariate_normal


# 初始化数据
def initDataSet():
    mean1 = (0, 0, 0)
    cov1 = np.diag((1, 1, 1))
    data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=200)
    mean2 = (2, 3, 4)
    cov2 = np.diag((2, 2, 2))
    data2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=500)
    data = np.vstack((data1, data2))
    return data



# 训练
def train(x, max_iter=100):
    m, n = np.shape(x)
    mu1 = x.min(axis=0)
    mu2 = x.max(axis=0)
    sigma1 = np.identity(n)
    sigma2 = np.identity(n)
    pi = 0.5
    #E步
    for i in range(max_iter):
        norm1 = multivariate_normal(mu1, sigma1)
        norm2 = multivariate_normal(mu2, sigma2)
        tau1 = pi * norm1.pdf(x)
        tau2 = (1 - pi) * norm2.pdf(x)
        w = tau1 / (tau1 + tau2)
        #M步
        mu1 = np.dot(w, x) / np.sum(w)
        mu2 = np.dot(1 - w, x) / np.sum(1 - w)
        sigma1 = np.dot(w * (x - mu1).T, (x - mu1)) / np.sum(w)
        sigma2 = np.dot((1 - w) * (x - mu2).T, (x - mu2)) / np.sum(1 - w)
        pi = np.sum(w) / m
    return (pi, mu1, mu2, sigma1, sigma2)


if __name__ == '__main__':
    data = initDataSet()
    pi, mu1, mu2, sigma1, sigma2 = train(data, 100)
    x = np.array([2, 2, 2])  # 测试样本
    norm1 = multivariate_normal(mu1, sigma1)  # 概率密度函数1
    norm2 = multivariate_normal(mu2, sigma2)
    p1 = pi * norm1.pdf(x)
    p2 = (1 - pi) * norm2.pdf(x)
    if p1 > p2:
        print "类别为第一类"
    else:
        print "类别为第二类"







