# -*- coding: utf-8 -*-
#
# reference: https://towardsdatascience.com/different-types-of-regularization-on-neuronal-network-with-pytorch-a9d6faf4793e

import numpy as np
from abc import ABCMeta, abstractmethod


class Regularizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def add_param(self, param):
        raise NotImplementedError()


class L1Regularizer(Regularizer):
    """
    L1 regularized loss
    """
    def __init__(self, lambda_reg=0.01):
        super(L1Regularizer, self).__init__()
        self.lambda_reg = lambda_reg

    def add_param(self, param):
        loss_reg = self.lambda_reg * param.abs().sum()
        return loss_reg


class L2Regularizer(Regularizer):
    """
    L2 regularized loss
    """
    def __init__(self, lambda_reg=0.01):
        super(L2Regularizer, self).__init__()
        self.lambda_reg = lambda_reg

    def add_param(self, param):
        loss_reg = self.lambda_reg * param.pow(2).sum()
        return loss_reg


class ElasticNetRegularizer(Regularizer):
    """
    Elastic Net Regularizer
    """
    def __init__(self, lambda_reg=0.01, alpha_reg=0.01):
        super(ElasticNetRegularizer, self).__init__()
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

    def add_param(self, param):
        loss_reg = self.lambda_reg * \
                   ((1 - self.alpha_reg) * param.pow(2).sum() +
                    self.alpha_reg * param.abs().sum())
        return loss_reg


class GroupLassoRegularizer(Regularizer):
    """
    GroupLasso Regularizer:
    The first dimension represents the input layer and
    the second dimension represents the output layer.
    The groups are defined by the column in the matrix W
    """
    def __init__(self, lambda_reg=0.01, group=0):
        super(GroupLassoRegularizer, self).__init__()
        self.lambda_reg = lambda_reg
        self.group = group

    def add_param(self, param):
        if len(param.shape) == 1:
            groupid = np.unique(self.group)
            NG = len(groupid)
            loss_group = 0
            for g in range(NG):
                index_ig = np.argwhere(self.group == groupid[g])
                loss_group += param[index_ig].norm(2, dim=0)
            loss_reg = self.lambda_reg * loss_group
        else:
            loss_reg = self.lambda_reg * param.norm(2, dim=0).sum()
        return loss_reg


class GroupSparseLassoRegularizer(Regularizer):
    """
    Group Sparse Lasso Regularizer
    """
    def __init__(self, lambda_reg=0.01):
        super(GroupSparseLassoRegularizer, self).__init__()
        self.lambda_reg = lambda_reg
        self.reg_l1 = L1Regularizer(lambda_reg=self.lambda_reg)
        self.reg_l2_l1 = GroupLassoRegularizer(lambda_reg=self.lambda_reg)

    def regularized_param(self, param):
        loss_reg = self.lambda_reg * \
                            (self.reg_l1.add_param(param) +
                             self.reg_l2_l1.add_param(param))

        return loss_reg