# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-13 17:15:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-13 17:15:29
"""
from ..tree.kd_tree import KDTree


class KNeighborsBase(object):
    def __init__(self):
        self.k_neighbors = None
        self.tree = None

    def fit(self, X, y, k_neighbors=3):
        self.k_neighbors = k_neighbors
        self.tree = KDTree()
        self.tree.build_tree(X, y)

    def _predict(self, Xi):
        """[summary]

        Arguments:
            Xi {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        raise NotImplementedError

    def _back_track(self, row):
        raise NotImplementedError

    def _predict(self, row):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


def main():
    raise NotImplementedError


def test():
    X = [[1, 2, 10], [3, 5, 8], [4, 6, 2], [8, 7, 5]]
    y = [0, 1, 2, 3]
    knn = KNeighborsBase()
    knn.fit(X, y)
    # nd = knn.tree._predict([1, 2, 9])
    print(nd.val)


if __name__ == "__main__":
    test()
    # main()
