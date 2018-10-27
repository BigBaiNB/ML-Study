# encoding: utf-8

#数据准备
import numpy as np

def initDataSet():
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3]
    ])
    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    # 白，黑，白，白，黑
    Q = [0, 1, 0, 0, 1]
    return pi, A, B, Q

pi, A, B, Q = initDataSet()
#前向算法
def calc_alpha(pi, A, B, Q):
    alpha = np.zeros((len(Q), len(A)))
    n = np.shape(A)[0]
    T = np.shape(Q)[0]
    for i in range(n):
        alpha[0][i] = pi[i] * B[i][Q[0]]
    tmp = np.zeros(n)
    for t in range(1, T):
        for i in range(n):
            for j in range(n):
                tmp[j] = alpha[t - 1][j] * A[j][i]
            alpha[t][i] = np.sum(tmp) * B[i][Q[t]]
    return alpha

def calc_Q(alpha):
    # 计算最终概率值：
    p = 0
    for i in alpha[-1]:
        p += i
    print(p)



if __name__ == '__main__':
    alpha = calc_alpha(pi, A, B, Q)
    calc_Q(alpha)











