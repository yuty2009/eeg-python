# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


""" Compute Common Spatial Patterns
W:  the transform matrix of CSP whose columns are the weight vector
R1: the covariance matrix of one class n-by-n (number of channels)
    R1 = X*X'/trace(X*X'); normalized covariance matrix
    X is N-by-T matrix where N is the number of channels and T is the
    number of samples in time.
R2:	the covariance matrix of the other class with the same dimension
"""
def CSP1(R1, R2):
    R = R1 + R2
    V1, U1 = np.linalg.eig(R)
    P = np.dot(np.diag(V1**(-1/2)), U1.T)
    S1 = np.dot(np.dot(P, R1), P.T)
    # S2 = np.dot(P, np.dot(R2, P.T))
    V2, U2 = np.linalg.eig(S1)
    W = np.dot(P.T, U2)
    ind = np.argsort(V2)
    W = W[:, ind]
    return W


def CSP2(R1, R2):
    V, U = sp.linalg.eig(R1, R2)
    ind = np.argsort(V)
    W = U[:, ind]
    return W


""" Train CSP filters
Rs:    variance of the multichannel signal
label: class label -1, 1
dim:   number of eigenvectors used
"""
def trainCSP2(Rs, labels, dim):
    num_trials = Rs.shape[0]
    num_channels = Rs.shape[1]

    # sort to make sure that -1 corresponds to left
    cc = np.unique(labels)
    # cc = np.sort(np.unique(labels))
    count_c1 = 0
    count_c2 = 0
    cov_c1 = np.zeros([num_channels, num_channels])
    cov_c2 = np.zeros([num_channels, num_channels])
    for i in range(num_trials):
        c = labels[i]
        if c == cc[0]:
            cov_c1 += Rs[i]
            count_c1 += 1
        elif c == cc[1]:
            cov_c2 += Rs[i]
            count_c2 += 1

    cov_c1 = cov_c1/count_c1
    cov_c2 = cov_c2/count_c2

    W_full = CSP1(cov_c1,cov_c2)

    if dim < 0:
        W = W_full
    else:
        W = np.concatenate((W_full[:, 0:int(dim/2)], W_full[:, num_channels-int(dim/2):]), axis=1)

    return W
