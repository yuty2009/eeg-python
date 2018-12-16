# -*- coding: utf-8 -*-

import numpy as np
from .utils import *
from .optimizer import *

## Fisher's Linear Discriminant Analysis
# y: N by 1 labels
# X: N by P matrix, N observation of P dimensional feature vectors
# wd: weight decay coefficient
def FLDA(y, X, wd = 1e-4):
    N, P = X.shape

    index1 = np.argwhere(y==1)
    index2 = np.argwhere(y==-1)
    N1 = len(index1)
    N2 = len(index2)

    X1 = X[index1,:]
    X2 = X[index2,:]

    mu1 = np.squeeze(np.mean(X1, axis=0))
    mu2 = np.squeeze(np.mean(X2, axis=0))

    Sw = np.cov(np.transpose(X))

    b = np.dot(np.linalg.inv(Sw + wd*np.eye(P)), (mu1 - mu2).T)
    b0 = -np.dot(mu1 + mu2, b)/2

    return b, b0


## Logistic regression for binary classification (Page 205-208 of PRML)
# Iterative reweighted least square (IRLS) by Newton-Raphson
# iterative optimization scheme.
# w_new = w_old - (PHI'*R*PHI)^(-1)*PHI'*(y-t);
#
# X: N by P design matrix with N samples of M features
# y: N by 1 target values {0,1}
# wd: weight decay coefficient
# b: P by 1 weight vector
# b0: bias
def logistic(y, X, wd = 1e-4):
    # add a constant column to cope with bias
    PHI = np.concatenate((np.ones([X.shape[0],1]),X),axis=1)
    N, P = PHI.shape
    y[y==-1] = 0 # the class label should be 1 or 0
    # initialization
    w = np.zeros(P) # rough initialization
    w[0] = np.log(np.mean(y)/(1-np.mean(y)))
    # stop conditions
    d_w = np.Inf
    maxit = 500
    stopeps = 1e-6

    i = 1
    while (d_w > stopeps) and (i < maxit) :
        wold = w

        t = 1/(1+np.exp(-np.dot(PHI,w))) # predicted target value
        R = np.diag(np.squeeze(t*(1-t))) # the variance matrix of target value
        # update with a norm2 regularization of w
        # H = PHI'*R*PHI + wd*eye(P);
        if (P > N):
            invH = woodburyinv(wd*np.eye(P), PHI.T, PHI, R)
        else:
            invH = np.linalg.inv(wd*np.eye(P) + np.dot(np.dot(PHI.T,R),PHI))

        w = w - np.dot(invH, np.dot(PHI.T,t-y) + wd*w)
        d_w = np.linalg.norm(wold-w)

        print('Iteration %d: wchange = %f' % (i, d_w))
        i = i + 1

    if (i >= maxit):
        print('Optimization finished with maximum iterations = %d' % maxit)

    return w[1:], w[0]


## Logistic regression for binary classification using SGD algorithm
# X: N by P design matrix with N samples of M features
# y: N by 1 target values {0,1}
# wd: weight decay coefficient
# b: P by 1 weight vector
# b0: bias
# need package pylbfgs installed
def logistic_sgd(y, X, wd=1e-4, optimizer=LbfgsOptimizer()):
    ##  min sum(log(1 + exp(-t.*(PHI * W)))) + wd *norm(w)
    def _logisticCost(w, *args):
        wd, PHI, y = args
        y = np.squeeze(y)
        z = y * np.matmul(PHI, w)
        t = 1 / (1 + np.exp(-z))
        grad = np.matmul(-PHI.T, y * (1 - t)) + wd * w
        cost = -np.sum(np.log(t)) + 0.5 * wd * np.dot(w.T, w)
        return cost, grad

    # add a constant column to cope with bias
    PHI = np.concatenate((np.ones([X.shape[0],1]),X),axis=1)
    N, P = PHI.shape
    # y[y==-1] = 0 # the class label should be 1 or 0
    # initialization
    w = np.zeros(P) # rough initialization

    # cost, grad = _logisticCost(w, wd, PHI, y)
    # grad1 = numgrad(_logisticCost, w, wd, PHI, y)
    # diff = np.linalg.norm(grad1 - grad) / np.linalg.norm(grad1 + grad)
    # print(diff)

    w = optimizer.minimize(_logisticCost, w, args=(wd, PHI, y))
    return w[1:], w[0]


## softmax activation function
def _softmax(X):
    expvx = np.exp(X - np.max(X, axis=1)[..., np.newaxis])
    return expvx / np.sum(expvx, axis=1, keepdims=True)

## Softmax regression using stocastic gradient descent algorithm
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 class labels (t=k indicate belong to class k)
# wd: weight decay coefficient
# W: P by K regression coefficients
def softmax_train(y, X, wd=1e-4, optimizer=LbfgsOptimizer()):
    ## Cross entropy error cost function
    def _softmaxCost(theta, *args):
        wd, PHI, y = args
        N, P = PHI.shape
        W = np.reshape(theta, [P,-1])
        t = _softmax(np.matmul(PHI, W))
        grad = (1./N)*np.matmul(PHI.T,t-y) + wd*W
        grad = grad.flatten()
        cost = -(1./N)*np.dot(y.flatten().T,np.log(t.flatten())) + 0.5*wd*np.sum(W.flatten()**2)
        return cost, grad

    K = len(np.unique(y))
    if len(y.shape) == 1 or y.shape[1] == 1:
        y = onehot(y, K)
    # add a constant column to cope with bias
    PHI = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
    N, P = PHI.shape
    W = np.ones([P, K])  # rough initialization
    theta = W.flatten()

    # cost, grad = _softmaxCost(theta, wd, PHI, y)
    # grad1 = numgrad(_softmaxCost, theta, wd, PHI, y)
    # diff = np.linalg.norm(grad1 - grad) / np.linalg.norm(grad1 + grad)
    # print(diff)

    opttheta = optimizer.minimize(_softmaxCost, theta, args=(wd, PHI, y))
    W = np.reshape(opttheta, W.shape)
    return W

def softmax_predict(X, W):
    PHI = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
    t = _softmax(np.matmul(PHI, W))
    y = np.argmax(t, axis=1)
    return y, t


if __name__ == "__main__":

    X = np.random.rand(5, 50)
    y = np.array([0, 1, 2, 3, 4])
    W = softmax_train(y, X)