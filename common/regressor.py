# -*- coding: utf-8 -*-

import numpy as np


## Ridge regression
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# b: P by 1 regression coefficients
# b0: the intercept
def ridgereg(y, X, coeff = 1e-4):
    N, P = X.shape
    PHI = np.concatenate((np.ones([N,1]), X), axis=1)
    invC = np.linalg.inv(coeff*np.eye(P+1)+ np.matmul(PHI.T, PHI))
    w = np.matmul(np.matmul(invC, PHI.T), y)
    b = w[1:]
    b0 = w[0]
    return b, b0