# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-14 15:34:28
@Last Modified by:   tushushu
@Last Modified time: 2018-08-14 15:34:28
"""
from .knn_base import KNeighborsBase


class KNeighborsClassifier(KNeighborsBase):
    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1D list with int or float

        Returns:
            float -- prediction of yi
        """

        heap = self._knn_search(Xi)
        return sum(nd.split[1] for nd in heap._items) / self.k_neighbors

    def predict(self, X, threshold=0.5):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Keyword Arguments:
            threshold {float} -- Prediction = 1 when probability >= threshold
            (default: {0.5})

        Returns:
            list -- 1d list object with float
        """

        return [int(self._predict(Xi) >= threshold) for Xi in X]
