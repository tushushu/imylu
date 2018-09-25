# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-15 11:19:44
@Last Modified by: tushushu
@Last Modified time: 2018-06-15 11:19:44
The paper links:
http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf
"""

from math import log
from ..tree.isolation_tree import IsolationTree


class IsolationForest(object):
    def __init__(self):
        """IsolationForest, randomly build some IsolationTree instance,
        and the average score of each IsolationTree


        Attributes:
        trees {list} -- 1d list with IsolationTree objects
        ajustment {float}
        """

        self.trees = None
        self.adjustment = None  # TBC

    def fit(self, X, n_samples=100, max_depth=10, n_trees=256):
        """Build IsolationForest with dataset X

        Arguments:
            X {list} -- 2d list with int or float

        Keyword Arguments:
            n_samples {int} -- According to paper, set number of samples
            to 256 (default: {256})
            max_depth {int} -- Tree height limit (default: {10})
            n_trees {int} --  According to paper, set number of trees
            to 100 (default: {100})
        """

        self.adjustment = self._get_adjustment(n_samples)
        self.trees = [IsolationTree(X, n_samples, max_depth)
                      for _ in range(n_trees)]

    def _get_adjustment(self, node_size):
        """Calculate adjustment according to the formula in the paper.

        Arguments:
            node_size {int} -- Number of leaf nodes

        Returns:
            float -- ajustment
        """

        if node_size > 2:
            i = node_size - 1
            ret = 2 * (log(i) + 0.5772156649) - 2 * i / node_size
        elif node_size == 2:
            ret = 1
        else:
            ret = 0
        return ret

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        # Calculate average score of xi at each tree
        score = 0
        n_trees = len(self.trees)
        for tree in self.trees:
            depth, node_size = tree._predict(Xi)
            score += (depth + self._get_adjustment(node_size))
        score = score / n_trees
        # Scale
        return 2 ** -(score / self.adjustment)

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        return [self._predict(Xi) for Xi in X]
