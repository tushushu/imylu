# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-14 15:34:28
@Last Modified by:   tushushu
@Last Modified time: 2018-08-14 15:34:28
"""
from .knn_base import KNeighborsBase
from typing import List


class KNeighborsClassifier(KNeighborsBase):
    def __init__(self):
        KNeighborsBase.__init__(self)

    def predict_prob(self, X: List)->List:
        """[summary]

        Args:
            X {list} -- 2d list object with int or float

        Returns:
            List: 1d list object with int or float
        """
        return [self._predict(Xi) for Xi in X]

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1D list with int or float

        Returns:
            float -- prediction of yi
        """

        heap = self._knn_search(Xi)
        n_pos = sum(nd.split[1] for nd in heap._items)
        return n_pos / self.k_neighbors

    def predict(self, X, threshold=0.5):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Keyword Arguments:
            threshold {float} -- (default: {0.5})

        Returns:
            list -- 1d list object with int or float
        """

        return [int(yi > threshold) for yi in self.predict_prob(X)]
