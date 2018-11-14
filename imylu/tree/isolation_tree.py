# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-24 13:59:41
@Last Modified by:   tushushu
@Last Modified time: 2018-08-24 13:59:41
The paper links:
http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf
"""
from random import sample, random, choice
from ..utils.utils import list_split


class Node(object):
    def __init__(self, size):
        """Node class to build tree leaves

        Attributes:
            size {int} -- Node size (default: {None})
            left {Node} -- Left child node
            right {Node} -- Right child node
            feature {int} -- Column index
            split {int} --  Split point
        """

        self.size = size

        self.left = None
        self.right = None
        self.feature = None
        self.split = None


class IsolationTree(object):
    def __init__(self, X, n_samples, max_depth):
        """Isolation Tree class

        Arguments:
            X {list} -- 2d list with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum depth of isolation tree
        """
        self.depth = 1
        # In case of n_samples is greater than n
        n = len(X)
        if n_samples > n:
            n_samples = n
        # Root node
        self.root = Node(n_samples)
        # Build isolation tree
        self._build_tree(X, n_samples, max_depth)

    def _get_split(self, X, idx, feature):
        """Randomly choose a split point

        Arguments:
            X {list} -- 2d list object with int or float
            idx {list} -- 1d list object with int
            feature {int} -- Column index of X

        Returns:
            int -- split point
        """

        # The split point should be greater than min(X[feature])
        unique = set(map(lambda i: X[i][feature], idx))
        # Cannot split
        if len(unique) == 1:
            return None
        unique.remove(min(unique))
        x_min, x_max = min(unique), max(unique)
        # Caution: random() -> x in the interval [0, 1).
        return random() * (x_max - x_min) + x_min

    def _build_tree(self, X, n_samples, max_depth):
        """The current node data space is divided into 2 sub space: less than
        the split point in the specified dimension on the left child of the
        current node, put greater than or equal to split point data on the
        current node's right child. Recursively construct new child nodes
        until the data cannot be splitted in the child nodes or the child
        nodes have reached the max_depth.

        Arguments:
            X {list} -- 2d list object with int or float
            n_samples {int} -- Subsample size
            max_depth {int} -- Maximum depth of IsolationTree
        """

        # Dataset shape
        m = len(X[0])
        n = len(X)
        # Initialize depth, node
        # Randomly selected sample points into the root node of the tree
        idxs = sample(range(n), n_samples)
        que = [(self.depth + 1, self.root, idxs)]
        # BFS
        while que:
            depth, nd, idxs = que.pop(0)
            # Terminate loop if tree depth is more than max_depth
            if depth > max_depth:
                depth -= 1
                break
            # Stop split if X cannot be splitted
            feature = choice(range(m))
            split = self._get_split(X, idxs, feature)
            if split is None:
                continue
            # Split
            idxs_split = list_split(X, idxs, feature, split)
            # Update properties of current node
            nd.feature = feature
            nd.split = split
            nd.left = Node(len(idxs_split[0]))
            nd.right = Node(len(idxs_split[1]))
            # Put children of current node in que
            que.append((depth + 1, nd.left, idxs_split[0]))
            que.append((depth + 1, nd.right, idxs_split[1]))
        # Update the depth of IsolationTree
        self.depth = depth

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1D list with int or float

        Returns:
            int -- the depth of the node which the xi belongs to
        """

        # Search Xi from the IsolationTree until Xi is at an leafnode
        nd = self.root
        depth = 0
        while nd.left and nd.right:
            if Xi[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
            depth += 1
        return depth, nd.size
