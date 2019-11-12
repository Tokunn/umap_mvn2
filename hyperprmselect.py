#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class EMP(object):

    def __init__(self):
        pass

    def get_edge(self, X):
        N = X.shape[0]
        k = int(5*np.log10(N))
        T = 0.1
        edge_idx = []
        for i in range(N):
            xij = self._knn(X[i], np.r_[X[:i], X[i+1:]], k)
            vij = (X[i].reshape(1, -1)-xij)/np.linalg.norm(X[i]-xij, axis=1).reshape(-1, 1)
            ni = np.sum(vij, axis=0)
            theta_ij = vij @ni.reshape(-1, 1)
            li = 0
            for j in range(k):
                if theta_ij[j] >= 0:
                    li += 1
            li /= k
            if li >= 1-T:
                edge_idx.append(i)
        return np.array(edge_idx)

    def _knn(self, z, X, k):
        dist = np.linalg.norm(X-z.reshape(1, -1), axis=1)
        dist_idx = np.argsort(dist)
        tmp = X[dist_idx]
        # print(dist[dist_idx])
        return tmp[:k]


class GenOutlier(object):
    def __init__(self):
        pass

    def gen_outlier(self, X, idx):
        N = X.shape[0]
        k = int(5*np.log10(N))

        sum_l = 0.0
        ni = []
        for i in idx:
            xij = self._knn(X[i], X, k)
            sum_l+=np.sum(np.linalg.norm(xij-X[i].reshape(1,-1), axis=1))/k
            n = np.sum((X[i].reshape(1,-1)-xij)/np.linalg.norm(X[i].reshape(1,-1)-xij).reshape(-1,1), axis=0)
            ni.append(n/np.linalg.norm(n))
        sum_l/=len(idx)

        ni = np.array(ni)
        return X[idx]+ni*sum_l

    def _knn(self, z, X, k):
        dist = np.linalg.norm(X-z.reshape(1,-1), axis=1)
        dist_idx = np.argsort(dist)
        tmp = X[dist_idx]
        return tmp[:k]


class GenTarget(object):
    def __init__(self):
        pass

    def gen_target(self, X):
        N = X.shape[0]
        k = int(5*np.log10(N))

        X_target = []
        for i in range(N):
            xij = self._knn(X[i], X, k)
            px = (-1)*np.sum((X[i].reshape(1,-1)-xij)/np.linalg.norm(X[i].reshape(1,-1)-xij).reshape(-1,1), axis=0)
            px /= np.linalg.norm(px)
            inner_p_xij = (xij-X[i].reshape(1,-1))@px.reshape(-1,1)
            pos_idx = np.where(inner_p_xij>0)
            xij, inner_p_xij = xij[pos_idx], inner_p_xij[pos_idx]
            min_val = np.min(inner_p_xij)
            X_target.append(X[i]+min_val*px)
        return np.array(X_target)

    def _knn(self, z, X, k):
        dist = np.linalg.norm(X-z.reshape(1,-1), axis=1)
        dist_idx = np.argsort(dist)
        tmp = X[dist_idx]
        return tmp[:k]


def makedata(X):
    emp = EMP()
    edge_idx = emp.get_edge(X)

    gen_o = GenOutlier()
    outlier_X = gen_o.gen_outlier(X, edge_idx)

    gen_t = GenTarget()
    target_X = gen_t.gen_target(X)

    return np.concatenate([outlier_X, target_X], axis=0), np.concatenate((np.zeros(len(outlier_X)), np.ones(len(target_X))))
