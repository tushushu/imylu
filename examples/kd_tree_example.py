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
from imylu.tree.kd_tree import KDTree


@run_time
def main():
    # Generate dataset randomly
    m = 3
    n = 10
    X = [[randint(0, 10) for _ in range(m)] for _ in range(n)]
    y = [randint(0, 1) for _ in range(n)]
    Xi = [randint(0, 10) for _ in range(m)]
    # Build KD Tree
    tree = KDTree()
    tree.build_tree(X, y)
    que = [(tree.root, None)]
    # Traverse KD Tree, show properties of nodes
    i = 0
    print("Traversing kd tree:\n")
    while que:
        nd, nd_father = que.pop(0)
        print("No.%d" % i, "| father node:", nd_father,
              "| feature:", nd.feature, "| split:", nd.split)
        if nd.left:
            que.append((nd.left, "No.%d" % i))
        if nd.right:
            que.append((nd.right, "No.%d" % i))
        i += 1
    print("Searching Xi:\n")
    nd = tree._search(Xi)
    print("Target:", Xi, "Match:", nd.split)


if __name__ == "__main__":
    main()
