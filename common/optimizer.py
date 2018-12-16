# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from abc import ABCMeta, abstractmethod

class GradientDescentOptimizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def minimize(self, fojb, x0, args):
        raise NotImplementedError()

class AdamOptimizer(GradientDescentOptimizer):
    def __init__(self, maxit=500, stopeps=1e-6):
        self.maxit = maxit
        self.stopeps = stopeps

    def minimize(self, fobj, x0, args):
        alpha = 0.01
        beta_1 = 0.9
        beta_2 = 0.999  # initialize the values of the parameters
        epsilon = 1e-8

        it = 0
        m_t = 0
        v_t = 0
        theta_0 = x0
        d_theta = np.Inf
        while (d_theta > self.stopeps) and (it < self.maxit):  # till it gets converged
            it = it + 1
            theta_prev = theta_0
            f_t, g_t = fobj(theta_0, *args)
            m_t = beta_1 * m_t + (1 - beta_1) * g_t  # updates the moving averages of the gradient
            v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)  # updates the moving averages of the squared gradient
            m_cap = m_t / (1 - (beta_1 ** it))  # calculates the bias-corrected estimates
            v_cap = v_t / (1 - (beta_2 ** it))  # calculates the bias-corrected estimates
            theta_0 = theta_0 - (alpha * m_cap) / (np.sqrt(v_cap) + epsilon)  # updates the parameters
            d_theta = np.linalg.norm(theta_0-theta_prev)
            print('Iteration %d: FuncValue = %f, d_theta = %f' % (it, f_t, d_theta))

        return theta_0


class LbfgsOptimizer(GradientDescentOptimizer):
    def __init__(self, maxit=500, stopeps=1e-5):
        self.maxit = maxit
        self.stopeps = stopeps

    def minimize(self, fobj, x0, args):
        theta, obj, info = fmin_l_bfgs_b(fobj, x0, args=args, maxiter=self.maxit, epsilon=self.stopeps, disp=1)
        return theta