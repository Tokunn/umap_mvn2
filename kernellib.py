#!/usr/bin/env python3

import numpy as np
from sklearn.metrics.pairwise import *


class UseKernel(object):
    def __init__(self, C=0.01, eta=0.99, pos=None, kernel=None, gamma='auto'):
        self.eta_flag = True
        if pos is not None:
            self.eta_flag = False
            self.pos = pos

        self.kernel_flag = False
        if kernel is not None:
            self.kernel_flag = True
            self.kernel = kernel
            self.gamma = gamma
            """
            gammaのauto機能はsklearnに合わせる. ただしsklearnではscaleと同等
            gamma : float, optional (default='auto')
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            Current default is 'auto' which uses 1 / n_features,
            if ``gamma='scale'`` is passed then it uses 1 / (n_features * X.var())
            as value of gamma. The current default of gamma, 'auto', will change
            to 'scale' in version 0.22. 'auto_deprecated', a deprecated version of
            'auto' is used as a default indicating that no explicit value of gamma
            was passed.
            """
        self.C = C
        self.eta = eta

    def _set_kernel(self, X, Y=None):
        if(self.kernel == 'rbf'):
            return rbf_kernel(X, Y, gamma=self.gamma)
        elif(self.kernel == 'poly'):
            return polynomial_kernel(X, Y, degree=self.d, gamma=1, coef0=1)
        else:
            return linear_kernel(X, Y)

    def _select_eig(self, e_val, e_vec):
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        if self.eta_flag is True:
            zero_idx = np.where(e_val > 0)
            e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
            sum_all = np.sum(e_val)
            sum_value = np.array(
                [np.sum(e_val[:i])/sum_all for i in range(1, len(e_val)+1)])
            r = int(np.min(np.where(sum_value >= self.eta)[0])+1)
        else:
            r = self.pos
        return e_vec.T[:r].T, e_val[:r]

    def _gen_emp(self, X):
        self.train_X = X
        if self.gamma == 'auto':
            self.gamma = 1/(self.train_X.shape[1]*self.train_X.var())
        # 標本特徴空間の生成

        train_K = self._set_kernel(self.train_X)
        e_val, e_vec = np.linalg.eigh(train_K)

        idx = np.where(e_val > 0)
        e_val = e_val[idx]
        e_vec = e_vec.T[idx].T

        self.L = np.sqrt(np.diag(1/e_val))
        self.P = e_vec

    def _map_emp(self, X):
        K = self._set_kernel(X, self.train_X)
        return (self.L@self.P.T@K.T).T

    def fit(self, train_X):  # train_Y ={+1 or -1}
        # gemerate the empirical feature space
        self._gen_emp(train_X)

    def transform(self, X):
        H = self._map_emp(X)
        return H
