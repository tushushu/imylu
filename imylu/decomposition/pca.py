"""
@Author: tushushu
@Date: 2019-01-04 15:35:11
"""
from numpy.linalg import eig
import heapq
import numpy as np


class PCA(object):
    """Principal component analysis (PCA)

    Arguments:
        n_components {int} -- Number of components to keep.
        eigen_vectors {array} -- The eigen vectors according to
        top n_components large eigen values.
    """

    def __init__(self):
        self.n_components = None
        self.eigen_vectors = None
        self.X_mean = None

    def _normalize(self, X):
        """Normalize the data by mean subtraction.

        Arguments:
            X {array} -- m * n array with int or float.

        Returns:
            array -- m * n array with int or float.
            array -- 1 * m array with int or float.
        """

        X_mean = X.mean(axis=0)
        return X - X_mean, X_mean

    def _get_covariance(self, X):
        """Calculate the covariance matrix.

        Arguments:
            X {array} -- m * n array with int or float.

        Returns:
            array -- n * n array with int or float.
        """

        m = X.shape[0]
        return X.T.dot(X) / (m - 1)

    def _get_top_eigen_vectors(self, X, n_components):
        """The eigen vectors according to top n_components large eigen values.

        Arguments:
            X {array} -- n * n array with int or float.
            n_components {int} -- Number of components to keep.

        Returns:
            array -- n * k array with int or float.
        """

        # Calculate eigen values and eigen vectors of covariance matrix.
        eigen_values, eigen_vectors = eig(X)
        # The indexes of top n_components large eigen values.
        indexes = heapq.nlargest(n_components, enumerate(eigen_values),
                                 key=lambda x: x[1])
        indexes = [x[0] for x in indexes]
        return eigen_vectors[:, indexes]

    def fit(self, X, n_components):
        """Fit the model with X.

        Arguments:
            X {array} -- m * n array with int or float.
            n_components {int} -- Number of components to keep.
        """

        X_norm, self.X_mean = self._normalize(X)
        X_cov = self._get_covariance(X_norm)
        self.n_components = n_components
        self.eigen_vectors = self._get_top_eigen_vectors(X_cov, n_components)

    def transform(self, X):
        """Apply the dimensionality reduction on X.

        Arguments:
            X {array} -- m * n array with int or float.

        Returns:
            array -- n * k array with int or float.
        """

        return (X - self.X_mean).dot(self.eigen_vectors)

    def fit_trasform(self, X, n_components):
        """Fit the model with X and apply the dimensionality reduction on X.

        Arguments:
            X {array} -- m * n array with int or float.
            n_components {int} -- Number of components to keep.

        Returns:
            array -- n * k array with int or float.
        """

        self.fit(X, n_components)
        return self.transform(X)
