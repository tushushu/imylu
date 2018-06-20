# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-15 11:19:44
@Last Modified by: tushushu
@Last Modified time: 2018-06-15 11:19:44
"""

from random import sample, random, choice
from math import ceil, log


class Node(object):
    def __init__(self, idx=None, features=None):
        """Node class to build tree leaves

        Keyword Arguments:
            idx {list} -- 1d list with int (default: {None})
        """
        # Sample indexes of dataset
        self.idx = idx
        # Feature indexes of dataset
        self.features = features
        # Feature to split
        self.feature = None
        # Split point
        self.split = None
        # Left child node
        self.left = None
        # Right child node
        self.right = None


class IsolationTree(object):
    def __init__(self, X, n_samples, max_depth):
        """Isolation Tree class

        Arguments:
            X {list} -- 2d list with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum height of isolation tree
        """
        self.height = 0
        # Randomly selected sample points into the root node of the tree
        n = len(X)
        m = len(X[0])
        idx = sample(range(n), n_samples)
        features = list(range(m))
        self.root = Node(idx, features)
        # Build isolation tree
        self._build_tree(X, max_depth)

    def _get_feature(self, X, nd):
        """Randomly choose a feature from X

        Arguments:
            X {list} -- 2d list object with int or float
            nd {node class} -- Current node to split

        Returns:
            int -- feature
        """

        # Filter out columns that cannot be splitted
        nd.features = [j for j in nd.features if len(set(
            map(lambda i: X[i][j], nd.idx))) > 1]
        # Randomly choose a feature from X
        if nd.features == []:
            return None
        else:
            return choice(nd.features)

    def _get_split(self, X, nd):
        """Randomly choose a split point

        Arguments:
            X {list} -- 2d list object with int or float
            nd {node class} -- Current node to split

        Returns:
            int -- split point
        """

        # The split point should be greater than min(X[feature])
        unique = set(map(lambda i: X[i][nd.feature], nd.idx))
        unique.remove(min(unique))
        x_min, x_max = min(unique), max(unique)
        # Caution: random() -> x in the interval [0, 1).
        return random() * (x_max - x_min) + x_min

    def _build_tree(self, X, max_depth):
        """The current node data space is divided into 2 sub space: less than the 
        split point in the specified dimension on the left child of the current node, 
        put greater than or equal to split point data on the current node's right child.
        Recursively construct new child nodes until the data cannot be splitted in the 
        child nodes or the child nodes have reached the max_depth.

        Arguments:
            X {list} -- 2d list object with int or float
            max_depth {int} -- Maximum depth of IsolationTree
        """

        # BFS
        que = [[0, self.root]]
        while que:
            depth, nd = que.pop(0)
            if depth == max_depth:
                break
            # Stop split if X cannot be splitted
            nd.feature = self._get_feature(X, nd)
            if nd.feature is None:
                continue
            nd.split = self._get_split(X, nd)
            idx_left = []
            idx_right = []
            # Split
            while nd.idx:
                i = nd.idx.pop()
                xi = X[i][nd.feature]
                if xi < nd.split:
                    idx_left.append(i)
                else:
                    idx_right.append(i)
            # Generate left and right child
            nd.left = Node(idx_left, nd.features)
            nd.right = Node(idx_right, nd.features)
            # Put the left and child into the que and depth plus one
            que.append([depth+1, nd.left])
            que.append([depth+1, nd.right])
        # Update the height of IsolationTree
        self.height = depth

    def _predict(self, row):
        """Auxiliary function of predict.

        Arguments:
            row {list} -- 1D list with int or float

        Returns:
            int -- the depth of the node which the row belongs to
        """

        # Search row from the IsolationTree until row is at an leafnode
        nd = self.root
        depth = 0
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
            depth += 1
        return depth, len(nd.idx)


class IsolationForest(object):
    def __init__(self):
        self.trees = None
        self.ajustment = None

    def fit(self, X, n_samples=10, max_depth=10, n_trees=100):
        n = len(X)
        self.ajustment = self._get_adjustment(n_samples, n)
        self.trees = [IsolationTree(X, n_samples, max_depth)
                      for _ in range(n_trees)]

    def _get_adjustment(self, n_samples, n):
        if n_samples > 2:
            i = n_samples - 1
            ret = 2 * (log(i) + 0.5772156649) - 2 * i / n
        elif n_samples == 2:
            ret = 1
        else:
            ret = 0
        return ret

    def _predict(self, row, n):
        score = 0
        n_trees = len(self.trees)
        for tree in self.trees:
            depth, nd_size = tree._predict(row)
            score += (depth + self._get_adjustment(nd_size, n))
        score = score / n_trees
        # Normalization
        return 2 ** -(score / self.ajustment)

    def predict(self, X):
        n = len(X)
        return [self._predict(row, n) for row in X]


if __name__ == "__main__":
    # # Generate a dataset randomly
    # n = 100
    # X = [[randint(0, n), randint(0, n)] for _ in range(n)]
    # # Add outliers
    # X.append([n*1000]*2)
    X = [[1+x/100 for _ in range(20)] for x in range(100)]
    X.append([1000 for _ in range(20)])
    clf = IsolationForest()
    clf.fit(X, n_samples=100, n_trees=100)
    for x, y in zip(X, clf.predict(X)):
        print(x, y)
    # print(clf.predict(X))
    # print(clf.trees[0].height)
    # print(clf.trees[0].root.feature, clf.trees[0].root.split)
    # print(clf.trees[0].root.left.feature, clf.trees[0].root.left.split)
    # print(clf.trees[0].root.right.feature, clf.trees[0].root.right.split)
    # for x in X:
    #     print(x, clf.trees[0]._predict(x))
    # clf = IsolationTree(X, n_samples=50, max_depth=50)
    # for row in X:
    #     print(clf._predict(row))
    # i = 49
    # n = 100
    # print(2 * (log(1) + 0.5772156649) - 2 * i / n)
