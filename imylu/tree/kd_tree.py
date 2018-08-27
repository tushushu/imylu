# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 19:19:52
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 19:19:52
"""
from ..utils import min_max_scale


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

    def _get_median_idx(self, X, idxs, feature):
        """Calculate the median of a column of data.

        Arguments:
            X {list} -- 2d list object with int or float
            idxs {list} -- 1D list with int
            feature {int} -- Feature number
            sorted_idxs_2d {list} -- 2D list with int

        Returns:
            list -- The row index corresponding to the median of this column.
        """

        n = len(idxs)
        # Ignoring the number of column elements is odd and even.
        k = n // 2
        # Get all the indexes and elements of column j as tuples
        col = map(lambda i: (i, X[i][feature]), idxs)
        # Sort the tuples by the values of elements and get the corresponding indexes.
        sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x: x[1]))
        # Search the median value
        median_idx = list(sorted_idxs)[k]
        return median_idx

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

    def _split_feature(self, X, idxs, feature, median_idx):
        """Split indexes into two arrays according to split point.

        Arguments:
            X {list} -- 2d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number
            median_idx {float} -- median index of the feature

        Returns:
            list -- [left idx, right idx]
        """

        idxs_split = [[], []]
        split_val = X[median_idx][feature]
        for idx in idxs:
            # Keep the split point in current node.
            if idx == median_idx:
                continue
            # Split
            xi = X[idx][feature]
            if xi < split_val:
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

        # Scale the data for calculating variances
        X_scale = min_max_scale(X)
        # Initialize with node, indexes
        nd = self.root
        idxs = range(len(X))
        que = [(nd, idxs)]
        # Breadth-First Search
        while que:
            nd, idxs = que.pop(0)
            n = len(idxs)
            # Stop split if there is only one element in this node
            if n == 1:
                nd.split = (X[idxs[0]], y[idxs[0]])
                continue
            # Split
            feature = self._choose_feature(X_scale, idxs)
            median_idx = self._get_median_idx(X, idxs, feature)
            idxs_left, idxs_right = self._split_feature(
                X, idxs, feature, median_idx)
            # Update properties of current node
            nd.feature = feature
            nd.split = (X[median_idx], y[median_idx])
            # Put children of current node in que
            if idxs_left != []:
                nd.left = Node()
                que.append((nd.left, idxs_left))
            if idxs_right != []:
                nd.right = Node()
                que.append((nd.right, idxs_right))

    def _search(self, Xi):
        # Search Xi from the KDTree until Xi is at an leafnode
        nd = self.root
        while nd.left and nd.right:
            if Xi[nd.feature] < nd.split[0][nd.feature]:
                nd = nd.left
            else:
                nd = nd.right
        return nd

    def _back_track(self, Xi, nd):
        raise NotImplementedError

    def search(self, Xi):
        raise NotImplementedError
