# -*- coding: utf-8 -*-

import numpy as np


## sigmoid activation function
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


## softmax activation function
def softmax(X):
    expvx = np.exp(X - np.max(X, axis=1)[..., np.newaxis])
    return expvx / np.sum(expvx, axis=1, keepdims=True)


## Compute numerical gradient for cost function J with respect to theta
# theta: a vector of parameters
# J: a function that outputs a real-number. Calling y = J(theta) will return the
# function value at theta.
def numgrad(J, theta, *args):
    epsilon = 1e-4
    # Initialize numgrad with zeros
    grad = np.zeros(theta.shape)
    for i in range(len(theta)):
        e = np.zeros(theta.shape)
        e[i] = 1
        costp, gradp = J(theta + e * epsilon, *args)
        costm, gradm = J(theta - e * epsilon, *args)
        grad[i] = (costp-costm)/(2*epsilon)
    return grad


## Generate a full matrix given the sparse representation
def sparse(ind1, ind2, values):
    m = np.max(ind1) + 1
    n = np.max(ind2) + 1
    A = np.zeros([m,n])
    for i in range(len(values)):
        A[ind1[i],ind2[i]] = values[i]
    return A


## Generate one-hot coded labels
def onehot(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_offset + labels.ravel()] = 1
    return labels_onehot


## Compute R square
# y: N by 1 labels
# X: N samples by [P1,P2,...,PN] features
def rsquare(y, X):
    dims = X.shape
    NF = np.prod(dims[1:]).astype(int)
    rr = np.zeros(NF)
    X1 = np.reshape(X, [dims[0], -1])
    for i in range(NF):
        rr[i] = np.correlate(y, X1[:,i])[0]**2
    if len(dims)>2: rr = np.reshape(rr,dims[1:])
    return rr


## Calculate the Woodbury identity
# (A + BD^{-1}C)^{-1} = A^{-1} - A^{-1}B(D+CA^{-1}B)^{-1}CA^{-1}
# which is useful when A is large and diagonal, and hence easy to invert,
# while B has many rows but few columns (and conversely for C) so that
# the right-hand side is much cheaper to evaluate than the left-hand side.
# We consider this condition only.
def woodburyinv(A,B,C,D):
    invA = np.diag(1/np.diag(A))
    # WD = invA - invA*B*(D+C*invA*B)^(-1)*C*invA;
    aa = np.matmul(invA,B)
    bb = np.linalg.inv(D+np.matmul(C,aa))
    cc = np.matmul(C,invA)
    WD = invA - np.matmul(np.matmul(aa,bb),cc)
    return WD


## Compute the eigenvalues of X'*X by SVD
# X is a N by P design matrix
def myeig(X):
    N, P = X.shape
    U, S, VH = np.linalg.svd(X)
    d1 = S**2
    M = min(N,P)
    d = np.zeros(P)
    d[:M] = d1[:M]
    return d


if __name__ == "__main__":
    a = np.random.rand(5)
    A = sparse(range(5), range(5), a)
    X = np.random.rand(5,10)
    D = np.eye(10)
    Z1 = woodburyinv(A, X, X.T, D)
    Z2 = np.linalg.inv(A+np.matmul(np.matmul(X,np.linalg.inv(D)),X.T))

    b = np.random.randn(4, 3)
    d1, v1 = np.linalg.eig(np.dot(a.T, a))
    d2 = myeig(a)

    pass