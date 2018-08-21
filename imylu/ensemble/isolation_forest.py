# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-15 11:19:44
@Last Modified by: tushushu
@Last Modified time: 2018-06-15 11:19:44
The paper links: http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf
"""

from random import sample, random, choice, randint
from math import ceil, log
from utils import run_time


class Node(object):
    def __init__(self, size):
        """Node class to build tree leaves

        Keyword Arguments:
            size {int} -- Node size (default: {None})
        """

        # Node size
        self.size = size
        # Feature to split
        self.split_feature = None
        # Split point
        self.split_point = None
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
        # In case of n_samples is greater than n
        n = len(X)
        if n_samples > n:
            n_samples = n
        # Root node
        self.root = Node(n_samples)
        # Build isolation tree
        self._build_tree(X, n_samples, max_depth)

    def _get_split(self, X, idx, split_feature):
        """Randomly choose a split point

        Arguments:
            X {list} -- 2d list object with int or float
            idx {list} -- 1d list object with int
            split_feature {int} -- Column index of X

        Returns:
            int -- split point
        """

        # The split point should be greater than min(X[feature])
        unique = set(map(lambda i: X[i][split_feature], idx))
        # Cannot split
        if len(unique) == 1:
            return None
        unique.remove(min(unique))
        x_min, x_max = min(unique), max(unique)
        # Caution: random() -> x in the interval [0, 1).
        return random() * (x_max - x_min) + x_min

    def _build_tree(self, X, n_samples, max_depth):
        """The current node data space is divided into 2 sub space: less than the
        split point in the specified dimension on the left child of the current node,
        put greater than or equal to split point data on the current node's right child.
        Recursively construct new child nodes until the data cannot be splitted in the
        child nodes or the child nodes have reached the max_depth.

        Arguments:
            X {list} -- 2d list object with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum depth of IsolationTree
        """

        # Dataset shape
        m = len(X[0])
        n = len(X)
        # Randomly selected sample points into the root node of the tree
        idx = sample(range(n), n_samples)
        # Depth, Node and idx
        que = [[0, self.root, idx]]
        # BFS
        while que and que[0][0] <= max_depth:
            depth, nd, idx = que.pop(0)
            # Stop split if X cannot be splitted
            nd.split_feature = choice(range(m))
            nd.split_point = self._get_split(X, idx, nd.split_feature)
            if nd.split_point is None:
                continue
            # Split
            idx_left = []
            idx_right = []
            while idx:
                i = idx.pop()
                xi = X[i][nd.split_feature]
                if xi < nd.split_point:
                    idx_left.append(i)
                else:
                    idx_right.append(i)
            # Generate left and right child
            nd.left = Node(len(idx_left))
            nd.right = Node(len(idx_right))
            # Put the left and child into the que and depth plus one
            que.append([depth+1, nd.left, idx_left])
            que.append([depth+1, nd.right, idx_right])
        # Update the height of IsolationTree
        self.height = depth

    def _predict(self, xi):
        """Auxiliary function of predict.

        Arguments:
            xi {list} -- 1D list with int or float

        Returns:
            int -- the depth of the node which the xi belongs to
        """

        # Search xi from the IsolationTree until xi is at an leafnode
        nd = self.root
        depth = 0
        while nd.left and nd.right:
            if xi[nd.split_feature] < nd.split_point:
                nd = nd.left
            else:
                nd = nd.right
            depth += 1
        return depth, nd.size


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
            n_samples {int} -- According to paper, set number of samples to 256 (default: {256})
            max_depth {int} -- Tree height limit (default: {10})
            n_trees {int} --  According to paper, set number of trees to 100 (default: {100})
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

    def _predict(self, xi):
        """Auxiliary function of predict.

        Arguments:
            xi {list} -- 1d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        # Calculate average score of xi at each tree
        score = 0
        n_trees = len(self.trees)
        for tree in self.trees:
            depth, node_size = tree._predict(xi)
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

        return [self._predict(xi) for xi in X]


@run_time
def main():
    print("Comparing average score of X and outlier's score...")
    # Generate a dataset randomly
    n = 100
    X = [[random() for _ in range(5)] for _ in range(n)]
    # Add outliers
    X.append([10]*5)
    # Train model
    clf = IsolationForest()
    clf.fit(X, n_samples=500)
    # Show result
    print("Average score is %.2f" % (sum(clf.predict(X)) / len(X)))
    print("Outlier's score is %.2f" % clf._predict(X[-1]))


if __name__ == "__main__":
    main()
