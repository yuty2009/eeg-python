# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.distributions as tfp
from bayesian.priors import *


class LinearModel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        torch.cuda.manual_seed(42) if cuda else torch.manual_seed(42)

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

    def __init__(self, sigma=1.0, verbose=False):
        super().__init__()
        self.sigma = sigma
        self.verbose = verbose
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        y = torch.tensor(y, device=self.device)

        # prior = tfp.Normal(loc=0.0, scale=1.0)
        # prior = LaplacePrior(mu=0, b=1.0)
        prior = GaussPrior(mu=0, sigma=1.0)
        # prior = GaussMixturePrior(mus=[0, 0], sigmas=[1.5, 0.1], pis=[0.5, 0.5])

        w_mu = nn.Parameter(torch.zeros((P, 1), device=self.device).normal_(0, 0.5))
        b_mu = nn.Parameter(torch.zeros(1, device=self.device).normal_(0, 0.5))
        w_rho = nn.Parameter(-3 * torch.ones_like(w_mu, device=self.device))
        b_rho = nn.Parameter(-3 * torch.ones_like(b_mu, device=self.device))

        parameters = [w_mu, w_rho, b_mu, b_rho]
        optimizer = optim.Adam(parameters, lr=0.08)

        maxsteps = 1500
        for step in range(maxsteps):

            w, b = sample_weights(w_mu, w_rho, b_mu, b_rho)
            yp = torch.mm(X, w) + b.expand(X.shape[0], -1)

            w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
            b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)

            loss_kl = kl_loss(prior, w, w_mu, w_sigma) + kl_loss(prior, b, b_mu, b_sigma)
            loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=self.sigma)
            loss = loss_mse + loss_kl

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if self.verbose:
                # print statistics every 100 steps
                if (step + 1) % 10 == 0:
                    print("Step [{}/{}], MSE Loss: {:.4f}, "
                          "KL Div: {:.4f} Total loss {:.4f}"
                          .format((step + 1), maxsteps,
                                  loss_mse.item(), loss_kl.item(), loss.item()))

        self.w = w_mu.cpu().detach().numpy()
        self.b = b_mu.cpu().detach().numpy()
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

    def __init__(self, sigma=1.0, verbose=False):
        super().__init__()
        self.sigma = sigma
        self.verbose = verbose
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        y = torch.tensor(y, device=self.device)

        prior_w_rho = nn.Parameter(torch.zeros(P, 1, device=self.device))
        prior_b = GaussPrior(0, 0.5)

        w_mu = nn.Parameter(torch.zeros((P, 1), device=self.device).normal_(0, 0.5))
        b_mu = nn.Parameter(torch.zeros(1, device=self.device).normal_(0, 0.5))
        w_rho = nn.Parameter(-3 * torch.ones_like(w_mu, device=self.device))
        b_rho = nn.Parameter(-3 * torch.ones_like(b_mu, device=self.device))

        parameters = [w_mu, w_rho, b_mu, b_rho, prior_w_rho]
        optimizer = optim.Adam(parameters, lr=0.08)

        maxsteps = 1500
        for step in range(maxsteps):

            w, b = sample_weights(w_mu, w_rho, b_mu, b_rho)
            yp = torch.mm(X, w) + b.expand(X.shape[0], -1)

            prior_w_sigma = 1e-6 + F.softplus(prior_w_rho, beta=1, threshold=20)
            prior_w = GaussPrior(mu=0, sigma=prior_w_sigma)
            w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
            b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)

            loss_kl = kl_loss(prior_w, w, w_mu, w_sigma) + \
                      kl_loss(prior_b, b, b_mu, b_sigma)
            loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=self.sigma)
            loss = loss_mse + loss_kl

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if self.verbose:
                # print statistics every 100 steps
                if (step + 1) % 10 == 0:
                    print("Step [{}/{}], MSE Loss: {:.4f}, "
                          "KL Div: {:.4f} Total loss {:.4f} "
                          "sum(prior_w_rho) {:.4f}"
                          .format((step + 1), maxsteps,
                                  loss_mse.item(), loss_kl.item(),
                                  loss.item(), prior_w_rho.sum().item()))

        self.w = w_mu.cpu().detach().numpy()
        self.b = b_mu.cpu().detach().numpy()
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

    def __init__(self, sigma=1.0, verbose=False):
        super().__init__()
        self.sigma = sigma
        self.verbose = verbose
        self.w, self.b = None, 0

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        y = torch.tensor(y, device=self.device)

        group = args
        if np.size(group) == 1:
            PG = P // group  # number of feature per-group
            group = np.arange(P) // PG + 1
        groupid = np.unique(group)
        NG = len(groupid)

        prior_w_rho = nn.Parameter(torch.zeros(NG, 1, device=self.device))
        prior_b = GaussPrior(0, 0.5)

        w_mu = nn.Parameter(torch.zeros((P, 1), device=self.device).normal_(0, 0.5))
        b_mu = nn.Parameter(torch.zeros(1, device=self.device).normal_(0, 0.5))
        w_rho = nn.Parameter(-3 * torch.ones_like(w_mu, device=self.device))
        b_rho = nn.Parameter(-3 * torch.ones_like(b_mu, device=self.device))

        parameters = [w_mu, w_rho, b_mu, b_rho, prior_w_rho]
        optimizer = optim.Adam(parameters, lr=0.08)

        maxsteps = 1500
        for step in range(maxsteps):

            w, b = sample_weights(w_mu, w_rho, b_mu, b_rho)
            yp = torch.mm(X, w) + b.expand(X.shape[0], -1)

            prior_w_rho_full = []
            for g in range(NG):
                index_ig = np.argwhere(group == groupid[g])
                prior_w_rho_full.append(prior_w_rho[g].expand(len(index_ig), 1))
            prior_w_rho_full = torch.cat(prior_w_rho_full)
            prior_w_sigma = 1e-6 + F.softplus(prior_w_rho_full, beta=1, threshold=20)
            prior_w = GaussPrior(mu=0, sigma=prior_w_sigma)
            w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
            b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)

            loss_kl = kl_loss(prior_w, w, w_mu, w_sigma) + \
                      kl_loss(prior_b, b, b_mu, b_sigma)
            loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=self.sigma)
            loss = loss_mse + loss_kl

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if self.verbose:
                # print statistics every 100 steps
                if (step + 1) % 10 == 0:
                    print("Step [{}/{}], MSE Loss: {:.4f}, "
                          "KL Div: {:.4f} Total loss {:.4f} "
                          "sum(prior_w_rho) {:.4f}"
                          .format((step + 1), maxsteps,
                                  loss_mse.item(), loss_kl.item(),
                                  loss.item(), prior_w_rho.sum().item()))

        self.w = w_mu.cpu().detach().numpy()
        self.b = b_mu.cpu().detach().numpy()
        return self.w, self.b

    def predict(self, X, args=None):
        assert self.w is not None, 'Please fit the model before use'
        yp = np.dot(X, self.w) + self.b
        return yp


def neg_log_likelihood(y_obs, y_pred, sigma=1.0):
    dist = GaussPrior(mu=y_pred, sigma=sigma)
    return -dist.log_prob(y_obs).sum()


def kl_loss(prior, w, mu, sigma):
    variational_dist = GaussPrior(mu, sigma)
    # variational_dist = tfp.Normal(mu, sigma)
    return torch.sum(variational_dist.log_prob(w) - prior.log_prob(w))


def sample_weights(w_mu, w_rho, b_mu, b_rho):
    """Quick method for sampling weights and exporting weights"""
    W_eps = w_mu.data.new(w_mu.size()).normal_()
    w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
    W = w_mu + 1 * w_sigma * W_eps

    if b_mu is not None:
        b_eps = b_mu.data.new(b_mu.size()).normal_()
        b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)
        b = b_mu + 1 * b_sigma * b_eps
    else:
        b = None

    return W, b


if __name__ == "__main__":
    pass
