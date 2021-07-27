# -*- coding: utf-8 -*-
#
# reference: https://github.com/JavierAntoran/Bayesian-Neural-Networks/
#            https://pytorch.org/docs/stable/distributions.html

import math
from numbers import Number
from abc import ABCMeta, abstractmethod


class BasePrior(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def log_prob(self, x):
        raise NotImplementedError()


class GammaPrior(BasePrior):
    def __init__(self, a, b, backend=None):
        self.backend = backend
        self.a = a
        self.b = b

    def log_prob(self, x):
        if self.backend is None:
            return (self.a * math.log(self.b) +
                    (self.a - 1) * x.log() -
                    self.b * x - self.a.lgamma())
        else:
            return (self.a * math.log(self.b) +
                    (self.a - 1) * self.backend.log(x) -
                    self.b * x - self.backend.lgamma(self.a))


class LaplacePrior(BasePrior):
    def __init__(self, mu, b):
        self.mu = mu
        self.b = b

    def log_prob(self, x):
        return -math.log(2 * self.b) - abs(x - self.mu) / self.b


class GaussPrior(BasePrior):
    def __init__(self, mu, sigma, backend=None):
        self.mu = mu
        self.sigma = sigma

        self.const_term = -(0.5) * math.log(2 * math.pi)
        if isinstance(self.sigma, Number):
            self.log_scale_term = math.log(self.sigma)
        else:
            if backend is None:
                self.log_scale_term = - self.sigma.log()
            else:
                self.log_scale_term = - backend.log(self.sigma)

    def log_prob(self, x):
        dist_term = - 0.5 * ((x - self.mu) / self.sigma) ** 2

        return self.const_term + self.log_scale_term + dist_term


class GaussMixturePrior(BasePrior):
    def __init__(self, mus, sigmas, pis, backend=None):
        self.backend = backend
        self.pis = pis
        self.components = []
        for mu, sigma in zip(mus, sigmas):
            component = GaussPrior(mu, sigma, backend=backend)
            self.components.append(component)

    def log_prob(self, x):
        ll_total = 0
        for component, pi in zip(self.components, self.pis):
            if self.backend is None:
                ll_total = ll_total + pi * component.log_prob(x).exp()
            else:
                ll_total = ll_total + pi * self.backend.exp(component.log_prob(x))

        ll_total = ll_total.log() if self.backend is None else self.backend.log(ll_total)

        return ll_total
