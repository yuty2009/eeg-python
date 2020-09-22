# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod


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


class SoftmaxClassifier(LinearModel):
    """ Softmax regression using stocastic gradient descent algorithm
    # X: N by P feature matrix, N number of samples, P number of features
    # y: N by 1 class labels (t=k indicate belong to class k)
    # wd: weight decay coefficient
    # W: P by K regression coefficients
    """

    def __init__(self, wd=1e-4, verbose=False):
        super().__init__()
        self.wd = wd
        self.verbose = verbose
        self.model = None

    def fit(self, X, y=None, args=None):
        N, P = X.shape
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        y = torch.tensor(y, device=self.device, dtype=torch.int64)

        K = len(torch.unique(y))
        self.model = nn.Sequential(nn.Linear(P, K))
        optimizer = optim.Adam(self.model.parameters(), lr=0.08)

        epochs = 200
        for epoch in range(epochs):
            yp = self.model(X)
            loss = F.cross_entropy(yp, y, reduction='sum')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.verbose:
                print("Epoch [{}/{}], Loss {:.4f}"
                      .format(epoch + 1, epochs, loss.item()))

    def predict(self, X, args=None):
        assert self.model is not None, 'Please fit the model before use'
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            t = self.model(X)
            y = t.argmax(dim=1)
        return y.cpu().detach().numpy(), t.cpu().detach().numpy()
