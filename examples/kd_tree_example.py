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

from random import randint
from time import time
from imylu.utils import gen_data, get_euclidean_distance
from imylu.tree.kd_tree import KDTree


def exhausted_search(tree, Xi):
    """[summary]

    Arguments:
        tree {KD Tree}
        Xi {list} -- [description]

    Returns:
        Node -- [description]
    """

    dist_best = float('inf')
    nd_best = None
    que = [tree.root]
    while que:
        nd = que.pop(0)
        dist = get_euclidean_distance(Xi, nd.split[0])
        if dist < dist_best:
            dist_best = dist
            nd_best = nd
        if nd.left is not None:
            que.append(nd.left)
        if nd.right is not None:
            que.append(nd.right)
    return nd_best


def main():
    test_times = 100
    run_time_1 = run_time_2 = 0
    for _ in range(test_times):
        # Generate dataset randomly
        low = 0
        high = 100
        n_rows = 10000
        n_cols = 3
        X = gen_data(low, high, n_rows, n_cols)
        y = gen_data(low, high, n_rows)
        Xi = gen_data(low, high, n_cols)

        # Build KD Tree
        tree = KDTree()
        tree.build_tree(X, y)

        # KD Tree Search
        start = time()
        nd1 = tree.nearest_neighbour_search(Xi)
        run_time_1 += time() - start
        ret1 = get_euclidean_distance(Xi, nd1.split[0])
        # Exhausted search
        start = time()
        nd2 = exhausted_search(tree, Xi)
        run_time_2 += time() - start
        ret2 = get_euclidean_distance(Xi, nd2.split[0])
        # Compare result
        assert ret1 == ret2, "target:%s\nrestult1:%s\nrestult2:%s\ntree:\n%s" % (
            str(Xi), str(nd1), str(nd2), str(tree))
    print("%d tests passed!" % test_times)
    print("KD Tree Search %.2f s" % run_time_1)
    print("Exhausted search %.2f s" % run_time_2)


if __name__ == "__main__":
    main()
