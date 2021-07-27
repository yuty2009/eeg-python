# -*- coding: utf-8 -*-

import numpy as np
from abc import ABCMeta, abstractmethod
import pymc3 as pm
import theano.tensor as tt


class LinearModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y=None, args=None):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X, args=None):
        raise NotImplementedError()


class BayesLinearRegression(LinearModel):
    """ Bayesian linear regression
    # X: N by P feature matrix, N number of samples, P number of features
    # y: N by 1 target vector
    # b: P by 1 regression coefficients
    # b0: the intercept
    """

    def __init__(self):
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        PHI = np.concatenate((np.ones([N, 1]), X), axis=1)

        with pm.Model() as model:
            alpha = pm.Uniform('alpha', lower=1e-9, upper=1e9, testval=2)
            weights = pm.Normal('weights', mu=0, sd=alpha**(-0.5), shape=P+1)
            mu = tt.dot(PHI, weights)
            beta = pm.Uniform('beta', lower=1e-9, upper=1e9, testval=10)
            predictions = pm.Normal('predictions', mu=mu, sd=beta**(-0.5), observed=y)

        map_estimate = pm.find_MAP(model=model)
        w = map_estimate['weights']

        self.w, self.b = w[1:], w[0]
        return self.w, self.b

    def predict(self, X, args=None):
        assert self.w is not None, 'Please fit the model before use'
        yp = np.dot(X, self.w) + self.b
        return yp


class BayesARDLinearRegression(LinearModel):
    """ Bayesian linear regression with ARD prior
    # refer to Page 347-348 of PRML book
    # X: N by P feature matrix, N number of samples, P number of features
    # y: N by 1 target vector
    # b: P by 1 regression coefficients
    # b0: the intercept
    """

    def __init__(self):
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        PHI = np.concatenate((np.ones([N, 1]), X), axis=1)

        with pm.Model() as model:
            alpha = pm.Uniform('alpha', lower=1e-9, upper=1e9, shape=P+1, testval=2*np.ones(P+1))
            tau = tt.diag(alpha**(-0.5))
            weights = pm.MvNormal('weights', mu=0, tau=tau, shape=P+1)
            mu = tt.dot(PHI, weights)
            beta = pm.Uniform('beta', lower=1e-9, upper=1e9, testval=10)
            predictions = pm.Normal('predictions', mu=mu, sd=beta**(-0.5), observed=y)

        map_estimate = pm.find_MAP(model=model)
        w = map_estimate['weights']

        self.w, self.b = w[1:], w[0]
        return self.w, self.b

    def predict(self, X, args=None):
        assert self.w is not None, 'Please fit the model before use'
        yp = np.dot(X, self.w) + self.b
        return yp


class BayesGARDLinearRegression(LinearModel):
    """ Bayesian linear regression with grouped ARD prior
    # X: N by P feature matrix, N number of samples, P number of features
    # y: N by 1 target vector
    # group: No. of groups or a group id vector
    #        e.g. group = [1 1 1 2 2 2 3 3 3 4 4 4]' indicates that there are
    #        4 group with 3 members in each
    # b: P by 1 regression coefficients
    # b0: the intercept
    """

    def __init__(self):
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        PHI = np.concatenate((np.ones([N, 1]), X), axis=1)

        group = args
        if np.size(group) == 1:
            PG = P // group  # number of feature per-group
            group = np.arange(P) // PG + 1
        group = np.append([0], group)  # account for bias
        groupid = np.unique(group)
        NG = len(groupid)

        with pm.Model() as model:
            alpha = pm.Uniform('alpha', lower=1, upper=1e9, shape=NG, testval=2*np.ones(NG))
            alpha_complete = np.array([None]*(P+1))
            for g in range(NG):
                index_ig = np.argwhere(group == groupid[g])
                alpha_complete[index_ig] = alpha[g]**(-0.5)
            tau = tt.diag(tt.stack(list(alpha_complete)))
            weights = pm.MvNormal('weights', mu=0, tau=tau, shape=P+1)
            mu = tt.dot(PHI, weights)
            beta = pm.Uniform('beta', lower=1e-9, upper=1e9, testval=10)
            predictions = pm.Normal('predictions', mu=mu, sd=beta**(-0.5), observed=y)

        map_estimate = pm.find_MAP(model=model)
        w = map_estimate['weights']

        self.w, self.b = w[1:], w[0]
        return self.w, self.b

    def predict(self, X, args=None):
        assert self.w is not None, 'Please fit the model before use'
        yp = np.dot(X, self.w) + self.b
        return yp


if __name__ == "__main__":
    pass
