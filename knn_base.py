# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-13 17:15:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-13 17:15:29
"""
from statistics import median
import copy


class Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None
        self.split = None
        self.val = None


class KDTree(object):
    def __init__(self):
        self.root = Node()

    def _get_variance(self, X, idxs, feature):
        """[summary]

        Arguments:
            X {[type]} -- [description]
            idxs {[type]} -- [description]
            feature {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        n = len(idxs)
        col_sum = col_sum_sqr = 0
        for idx in idxs:
            xi = X[idx][feature]
            col_sum += xi
            col_sum_sqr += xi ** 2
        return col_sum_sqr / n - (col_sum / n) ** 2

    def _choose_feature(self, X, idxs, min_variance=1e-5):
        """[summary]

        Arguments:
            X {[type]} -- [description]
            idxs {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        m = len(X[0])
        variances = map(lambda x: self._get_variance(X, idxs, x), range(m))
        variances = filter(lambda x: x > min_variance, variances)
        return max(enumerate(variances), default=None, key=lambda x: x[1])

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
        """[summary]

        Arguments:
            X {[type]} -- [description]
            y {[type]} -- [description]
        """

        n = len(X)
        nd = self.root
        idxs = range(n)
        que = [(nd, idxs)]
        while que:
            nd, idxs = que.pop(0)
            split_info = self._choose_feature(X, idxs)
            if split_info is None:
                nd.val = [(X[idx], y[idx]) for idx in idxs]
                continue
            nd.feature, _ = split_info
            split = median(X[idx][nd.feature] for idx in idxs)
            idxs_left, idxs_right = self._split_feature(
                X, idxs, nd.feature, split)
            nd.split = split
            nd.left = Node()
            nd.right = Node()
            que.append((nd.left, idxs_left))
            que.append((nd.right, idxs_right))

    def _get_leaves(self):
        """Gets all leaf nodes of a KD tree

        Returns:
            list -- 1D list with KDTree.Node objects
        """

        nodes = []
        que = [self.root]
        while que:
            node = que.pop(0)
            if node.left is None or node.right is None:
                nodes.append(node)
                continue
            left_node = node.left
            right_node = node.right
            que.append(left_node)
            que.append(right_node)
        return nodes


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

    def _search(self, row):
        nd = self.tree.root
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
    tree = KDTree()
    tree.build_tree(X, y)
    leaves = tree._get_leaves()
    for leave in leaves:
        print(leave, leave.val)


if __name__ == "__main__":
    test()
    # main()
