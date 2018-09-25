# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-09-14 14:58:54
@Last Modified by:   tushushu
@Last Modified time: 2018-09-14 14:58:54
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.utils import get_euclidean_distance
from imylu.neighbors.knn_base import KNeighborsBase


def exhausted_search(X, Xi, k):
    """Linear search the nearest neighbour.

    Arguments:
        X {list} -- 2d list with int or float.
        Xi {list} -- 1d list with int or float.
        k {int} -- number of neighbours.

    Returns:
        list -- The lists of the K nearest neighbour.
    """

    idxs = []
    for _ in range(k):
        best_dist = float('inf')
        idxs.append(None)
        for i, row in enumerate(X):
            dist = get_euclidean_distance(Xi, row)
            if dist < best_dist and i not in idxs:
                best_dist = dist
                idxs[-1] = i
    return [X[i] for i in idxs]


def main():
    X = [[1, 2], [3, 4], [5, 10], [15, 20]]
    y = [1, 1, 0, 0]
    Xi = [0, 0]
    k = 2

    ret1 = exhausted_search(X, Xi, k)
    print(ret1)

    model = KNeighborsBase()
    model.fit(X, y, k_neighbors=k)
    ret2 = model._knn_search(Xi)
    for x in ret2.items:
        print(x)


if __name__ == "__main__":
    main()
