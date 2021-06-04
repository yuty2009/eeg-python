# -*- coding: utf-8 -*-

import re
import math
import time
import string
import unicodedata
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
    num_labels = len(labels)
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_offset + labels.ravel()] = 1
    return labels_onehot


## Compute R square
# X: N samples by [P1,P2,...,PN] features
# y: N by 1 labels
def rsquare(X, y):
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


def calc_confusion_matrix(yt, yp, num_classes):
    confusion = np.zeros((num_classes, num_classes))
    for yt1, yp1 in zip(yt, yp):
        confusion[yt1][yp1] += 1
    # Normalize by dividing every row by its sum
    for i in range(num_classes):
        confusion[i] = confusion[i] / (np.sum(confusion[i]) + 1e-9)
    return confusion


def plot_matrix(mat, xlabels=None, ylabels=None):
    from matplotlib import pyplot as plt
    from matplotlib import ticker as ticker

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat)
    fig.colorbar(cax)

    # Set up axes
    xlabels = '' if xlabels is None else xlabels
    ylabels = '' if ylabels is None else ylabels
    ax.set_xticklabels([''] + xlabels, rotation=90)
    ax.set_yticklabels([''] + ylabels)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == "__main__":
    a = np.random.rand(5)
    A = sparse(range(5), range(5), a)
    X = np.random.rand(5,10)
    D = np.eye(10)
    Z1 = woodburyinv(A, X, X.T, D)
    Z2 = np.linalg.inv(A+np.matmul(np.matmul(X,np.linalg.inv(D)),X.T))

    b = np.random.randn(4, 3)
    d1, v1 = np.linalg.eig(np.dot(b.T, b))
    d2 = myeig(b)

    mat = np.random.randn(5,5)
    labels = ['one', 'two', 'three', 'four', 'five']
    plot_matrix(mat, labels, labels)
