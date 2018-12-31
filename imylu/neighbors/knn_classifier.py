# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-14 15:34:28
@Last Modified by:   tushushu
@Last Modified time: 2018-08-14 15:34:28
"""
from .knn_base import KNeighborsBase


class KNeighborsClassifier(KNeighborsBase):
    def __init__(self):
        KNeighborsBase.__init__(self)

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1D list with int or float

        Returns:
            float -- prediction of yi
        """

        heap = self._knn_search(Xi)
        n_pos = sum(nd.split[1] for nd in heap._items)
        return int(n_pos * 2 > self.k_neighbors)
