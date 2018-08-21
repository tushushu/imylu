# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-13 17:15:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-13 17:15:29
"""


class KNeighborsBase(object):
    def __init__(self):
        self.k_neighbors = None
        self.tree = None

    def _get_distance(self, arr1, arr2):
        """Calculate the Euclidean distance of two vectors

        Arguments:
            arr1 {list} -- 1d list object with int or float
            arr2 {list} -- 1d list object with int or float

        Returns:
            float -- Euclidean distance
        """

        return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2))

    def fit(self, X, y, k_neighbors=3):
        self.k_neighbors = k_neighbors
        self.tree = KDTree()
        self.tree.build_tree(X, y)

    def _predict(self, row):
        """[summary]

        Arguments:
            row {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        nd = self.root
        while nd.left is not None and nd.right is not None:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd

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
    nd = knn.tree.search([1, 2, 9])
    print(nd.val)


if __name__ == "__main__":
    test()
    # main()
