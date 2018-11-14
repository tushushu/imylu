# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-22 10:37:56
@Last Modified by:   tushushu
@Last Modified time: 2018-08-22 10:37:56
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.utils.kd_tree import KDTree
from imylu.utils.load_data import gen_data
from imylu.utils.utils import get_euclidean_distance
from time import time


def exhausted_search(X, Xi):
    """Linear search the nearest neighbour.

    Arguments:
        X {list} -- 2d list with int or float.
        Xi {list} -- 1d list with int or float.

    Returns:
        list -- 1d list with int or float.r.
    """

    dist_best = float('inf')
    row_best = None
    for row in X:
        dist = get_euclidean_distance(Xi, row)
        if dist < dist_best:
            dist_best = dist
            row_best = row
    return row_best


def main():
    print("Testing KD Tree...")
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

        # Build KD Tree
        tree = KDTree()
        tree.build_tree(X, y)

        # KD Tree Search
        start = time()
        nd = tree.nearest_neighbour_search(Xi)
        run_time_1 += time() - start
        ret1 = get_euclidean_distance(Xi, nd.split[0])

        # Exhausted search
        start = time()
        row = exhausted_search(X, Xi)
        run_time_2 += time() - start
        ret2 = get_euclidean_distance(Xi, row)

        # Compare result
        assert ret1 == ret2, "target:%s\nrestult1:%s\nrestult2:%s\ntree:\n%s" \
            % (Xi, nd, row, tree)
    print("%d tests passed!" % test_times)
    print("KD Tree Search %.2f s" % run_time_1)
    print("Exhausted search %.2f s" % run_time_2)


if __name__ == "__main__":
    main()
