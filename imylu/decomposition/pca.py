"""
@Author: tushushu
@Date: 2019-01-04 15:35:11
@Last Modified by: tushushu
@Last Modified time: 2019-05-03 09:13:34
"""
import heapq

from numpy import ndarray
from numpy.linalg import eig


class PCA:
    """Principal component analysis (PCA)

    Arguments:
        n_components {int} -- Number of components to keep.
        eigen_vectors {ndarray} -- The eigen vectors according to
        top n_components large eigen values.
    """

    def __init__(self):
        self.n_components = None
        self.eigen_vectors = None
        self.avg = None

    @staticmethod
    def _normalize(data: ndarray):
        """Normalize the data by mean subtraction.

        Arguments:
            data {ndarray} -- Training data.

        Returns:
            ndarray -- Normalized data with shape(n_rows, n_cols)
            ndarray -- Mean of data with shape(1, n_cols)
        """

        avg = data.mean(axis=0)
        return data - avg, avg

    @staticmethod
    def _get_covariance(data: ndarray) -> ndarray:
        """Calculate the covariance matrix of data.

        Arguments:
            data {ndarray} -- Training data.

        Returns:
            ndarray -- covariance matrix with shape(n_cols, n_cols)
        """

        n_rows = data.shape[0]
        return data.T.dot(data) / (n_rows - 1)

    @staticmethod
    def _get_top_eigen_vectors(data: ndarray, n_components: int) -> ndarray:
        """The eigen vectors according to top n_components large eigen values.

        Arguments:
            data {ndarray} -- Training data.
            n_components {int} -- Number of components to keep.

        Returns:
            ndarray -- eigen vectors with shape(n_cols, n_components).
        """

        # Calculate eigen values and eigen vectors of covariance matrix.
        eigen_values, eigen_vectors = eig(data)
        # The indexes of top n_components large eigen values.
        _indexes = heapq.nlargest(n_components, enumerate(eigen_values),
                                  key=lambda x: x[1])
        indexes = [x[0] for x in _indexes]
        return eigen_vectors[:, indexes]

    def fit(self, data: ndarray, n_components: int):
        """Fit the model with data.

        Arguments:
            data {ndarray} -- Training data.
            n_components {int} -- Number of components to keep.
        """

        data_norm, self.avg = self._normalize(data)
        data_cov = self._get_covariance(data_norm)
        self.n_components = n_components
        self.eigen_vectors = self._get_top_eigen_vectors(
            data_cov, n_components)

    def transform(self, data: ndarray) -> ndarray:
        """Apply the dimensionality reduction on X.

        Arguments:
            data {ndarray} -- Training data.

        Returns:
            ndarray -- with shape(n_cols, n_components).
        """

        return (data - self.avg).dot(self.eigen_vectors)

    def fit_trasform(self, data: ndarray, n_components: int) -> ndarray:
        """Fit the model with data and apply the dimensionality reduction on data.

        Arguments:
            data {ndarray} -- Training data.
            n_components {int} -- Number of components to keep.

        Returns:
            ndarray -- with shape(n_cols, n_components).
        """

        self.fit(data, n_components)
        return self.transform(data)
