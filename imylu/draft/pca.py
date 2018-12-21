"""
@Author: tushushu
@Date: 2018-12-20 14:08:28
"""


import numpy as np
from numpy.linalg import eig
from ..utils.matrix import Matrix


class PCA(object):
    def _scale(self, X, X_avg):
        """(m, n) matrix X substracts (1, n) matrix X_avg.

        Arguments:
            X {Matrix} -- 2d Matrix with int or float.
            X_avg {Matrix} -- 1d Matrix with int or float.

        Returns:
            Matrix -- 2d Matrix with int or float.
        """

        def f(Xi):
            return [a - b for a, b in zip(Xi, X_avg.data)]
        return Matrix([f(Xi) for Xi in X.data])

    def _get_covariance(self, X1, X2):
        """Calculate the covariance of X1 and X2.
        Cov(X, Y) = E((X - E(X)) * (Y - E(Y))) = E(X*Y) - E(X)E(Y)

        Arguments:
            X1 {list} -- 1d list with int or float.
            X2 {list} -- 1d list with int or float.

        Returns:
            float
        """

        assert len(X1) == len(X2), "Different lengths of X1 and X2!"
        m = len(X1)
        X12_avg = X1_avg = X2_avg = 0
        for a, b in zip(X1, X2):
            X12_avg += a * b / m
            X1_avg += a / m
            X2_avg += b / m
        return X12_avg - X1_avg * X2_avg

    def _get_covariance_matrix(self, X):
        """[summary]

        Arguments:
            X {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        n = X.shape[1]
        ret = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i <= j:
                    X1 = X.col(i)
                    X2 = X.col(j)
                    ret[i][j] = self._get_covariance(X1, X2)
                else:
                    ret[i][j] = ret[j][i]
        return Matrix(ret)

    def eig(self, X):
        raise NotImplementedError

    def get_top_k(self, arr):
        raise NotImplementedError

    def fit(self, X, k):
        # Standardize by remove average
        X = X - X.mean(axis=0)

        # Calculate covariance matrix:
        X_cov = np.cov(X.T, ddof=0)

        # Calculate  eigenvalues and eigenvectors of covariance matrix
        eigenvalues, eigenvectors = eig(X_cov)

        # top k large eigenvectors
        klarge_index = eigenvalues.argsort()[-k:][::-1]
        k_eigenvectors = eigenvectors[klarge_index]

        return np.dot(X, k_eigenvectors.T)
