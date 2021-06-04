# -*- coding: utf-8 -*-
#
# reference:
# https://github.com/pyRiemann/pyRiemann/

import numpy
from common.riemann.base import *


def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None):
    """Return the mean covariance matrix according to the Riemannian metric.
    The procedure is similar to a gradient descent minimizing the sum of
    riemannian distance to the mean.
    .. math::
            \mathbf{C} = \\arg\min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}  # noqa
    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the gradient descent. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample
    :returns: the mean covariance matrix
    """
    # init
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = numpy.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = numpy.finfo(numpy.float64).max
    crit = numpy.finfo(numpy.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = numpy.zeros((Ne, Ne))

        for index in range(Nt):
            tmp = numpy.dot(numpy.dot(Cm12, covmats[index, :, :]), Cm12)
            # J += sample_weight[index] * logm(tmp)
            J += logm(tmp)

        crit = numpy.linalg.norm(J, ord='fro')
        h = nu * crit
        C = numpy.dot(numpy.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C


def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space. according to
    the reference point Cref
    :param covmats: np.ndarray
        Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)
    """
    Nt, Ne, Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = numpy.triu_indices_from(Cref)
    Nf = int(Ne * (Ne + 1) / 2)
    T = numpy.empty((Nt, Nf))
    coeffs = (numpy.sqrt(2) * numpy.triu(numpy.ones((Ne, Ne)), 1) +
              numpy.eye(Ne))[idx]
    for index in range(Nt):
        tmp = numpy.dot(numpy.dot(Cm12, covmats[index, :, :]), Cm12)
        tmp = logm(tmp)
        T[index, :] = numpy.multiply(coeffs, tmp[idx])
    return T


def untangent_space(T, Cref):
    """Project a set of Tangent space vectors back to the manifold.
    :param T: np.ndarray
        the Tangent space , a matrix of Ntrials X (channels * (channels + 1)/2)
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        A set of Covariance matrix, Ntrials X Nchannels X Nchannels
    """
    Nt, Nd = T.shape
    Ne = int((numpy.sqrt(1 + 8 * Nd) - 1) / 2)
    C12 = sqrtm(Cref)

    idx = numpy.triu_indices_from(Cref)
    covmats = numpy.empty((Nt, Ne, Ne))
    covmats[:, idx[0], idx[1]] = T
    for i in range(Nt):
        triuc = numpy.triu(covmats[i], 1) / numpy.sqrt(2)
        covmats[i] = (numpy.diag(numpy.diag(covmats[i])) + triuc + triuc.T)
        covmats[i] = expm(covmats[i])
        covmats[i] = numpy.dot(numpy.dot(C12, covmats[i]), C12)

    return covmats
