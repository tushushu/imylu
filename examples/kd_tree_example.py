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
from imylu.utils import run_time
from imylu.neighbors.kd_tree import KDTree


@run_time
def main():
    m = 5
    n = 30
    X = [[randint(0, 10) for _ in range(m)] for _ in range(n)]
    y = [randint(0, 1) for _ in range(n)]
    tree = KDTree()
    tree.build_tree(X, y)
    que = [tree.root]
    i = 0
    print("Traversing kd tree:\n")
    while que:
        nd = que.pop(0)
        print("No.:", i, "| feature:", nd.feature, "| split:", nd.split)
        i += 1
        if nd.left:
            que.append(nd.left)
        if nd.right:
            que.append(nd.right)


if __name__ == "__main__":
    main()
