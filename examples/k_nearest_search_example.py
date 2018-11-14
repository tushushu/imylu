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

from imylu.neighbors.knn_base import KNeighborsBase
from imylu.utils.load_data import gen_data
from imylu.utils.utils import get_euclidean_distance
from time import time


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
            if i in idxs:
                continue
            dist = get_euclidean_distance(Xi, row)
            if dist < best_dist:
                best_dist = dist
                idxs[-1] = i
    return [X[i] for i in idxs]


def test():
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


def main():
    print("Testing K nearest search...")
    test_times = 100
    run_time_1 = run_time_2 = 0
    for _ in range(test_times):
        # Generate dataset randomly
        low = 0
        high = 100
        n_rows = 1000
        n_cols = 2
        X = gen_data(low, high, n_rows, n_cols)
        y = gen_data(low, high, n_rows)
        Xi = gen_data(low, high, n_cols)

        # Build KNN
        k = 2
        model = KNeighborsBase()
        model.fit(X, y, k_neighbors=k)

        # KD Tree Search
        start = time()
        heap = model._knn_search(Xi)
        run_time_1 += time() - start
        ret1 = [get_euclidean_distance(Xi, nd.split[0]) for nd in heap.items]
        ret1.sort()

        # Exhausted search
        start = time()
        ret2 = exhausted_search(X, Xi, k)
        run_time_2 += time() - start
        ret2 = [get_euclidean_distance(Xi, row) for row in ret2]
        ret2.sort()

        # Compare result
        assert ret1 == ret2, "target:%s\nrestult1:%s\nrestult2:%s\ntree:\n%s" \
            % (Xi, ret1, ret2, model.tree)

    print("%d tests passed!" % test_times)
    print("KNN Search %.2f s" % run_time_1)
    print("Exhausted search %.2f s" % run_time_2)


if __name__ == "__main__":
    main()
