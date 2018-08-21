# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 19:19:52
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 19:19:52
"""


class Node(object):
    def __init__(self):
        """Node class to build tree leaves.
        """

        self.left = None
        self.right = None
        self.feature = None
        self.split = None


class KDTree(object):
    """KDTree class to improve search efficiency in KNN.

    Attributes:
        root: the root node of KDTree
    """

    def __init__(self):
        self.root = Node()

    def _get_sorted_idxs(self, X):
        """Caculate sorted indexes of X.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 2D list with int
        """

        m = len(X[0])
        n = len(X)
        sorted_idxs_2d = []
        for j in range(m):
            col = map(lambda i: (i, X[i][j]), range(n))
            sorted_idxs_1d = list(map(
                lambda x: x[0], sorted(col, key=lambda x: x[1])))
            sorted_idxs_2d.append(sorted_idxs_1d)
        return sorted_idxs_2d

    def _get_median(self, X, idxs, feature, sorted_idxs_2d):
        """Calculate the median of a column of data.

        Arguments:
            X {list} -- 2d list object with int or float
            idxs {list} -- 1D list with int
            feature {int} -- Feature number
            sorted_idxs_2d {list} -- 2D list with int

        Returns:
            list -- The row corresponding to the median of this column.
        """

        # Ignoring the number of column elements is odd and even.
        k = len(idxs) // 2
        sorted_idxs_1d = list(map(lambda i: sorted_idxs_2d[i][feature], idxs))
        median_idx = sorted_idxs_1d[k]
        return X[median_idx]

    def _get_variance(self, X, idxs, feature):
        """Calculate the variance of a column of data.

        Arguments:
            X {list} -- 2d list object with int or float
            idxs {list} -- 1D list with int
            feature {int} -- Feature number

        Returns:
            float -- variance
        """

        n = len(idxs)
        col_sum = col_sum_sqr = 0
        for idx in idxs:
            xi = X[idx][feature]
            col_sum += xi
            col_sum_sqr += xi ** 2
        # D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2
        return col_sum_sqr / n - (col_sum / n) ** 2

    def _choose_feature(self, X, idxs):
        """Choose the feature which has maximum variance.

        Arguments:
            X {list} -- 2d list object with int or float
            idxs {list} -- 1D list with int

        Returns:
            feature number {int}
        """

        m = len(X[0])
        variances = map(lambda j: (
            j, self._get_variance(X, idxs, j)), range(m))
        return max(variances, key=lambda x: x[1])[0]

    def _split_feature(self, X, idxs, feature, split):
        """Split indexes into two arrays according to split point.

        Arguments:
            X {list} -- 2d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number
            split {float} -- Split point of the feature

        Returns:
            list -- [left idx, right idx]
        """

        idxs_split = [[], []]
        for idx in idxs:
            xi = X[idx][feature]
            if xi < split:
                idxs_split[0].append(idx)
            else:
                idxs_split[1].append(idx)
        return idxs_split

    def build_tree(self, X, y):
        """Build a kd tree.

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int or float
        """

        sorted_idxs_2d = self._get_sorted_idxs(X)
        # Initialize with node, indexes
        nd = self.root
        idxs = range(len(X))
        que = [(nd, idxs)]
        # Breadth-First Search
        while que:
            nd, idxs = que.pop(0)
            n = len(idxs)
            # Stop split if there is no element in this node
            if n == 0:
                continue
            # Stop split if there is only one element in this node
            if n == 1:
                nd.val = [(X[idx], y[idx]) for idx in idxs]
                continue
            # Split
            nd.feature = self._choose_feature(X, idxs)
            split = self._get_median(X, idxs, nd.feature, sorted_idxs_2d)
            idxs_left, idxs_right = self._split_feature(
                X, idxs, nd.feature, split)
            nd.split = split
            nd.left = Node()
            nd.right = Node()
            que.append((nd.left, idxs_left))
            que.append((nd.right, idxs_right))
