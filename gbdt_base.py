# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-13 10:29:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-13 10:29:29
"""
from regression_tree import RegressionTree
from random import choices


class GradientBoostingBase(object):
    def __init__(self):
        """GBDT base class.
        http://statweb.stanford.edu/~jhf/ftp/stobst.pdf

        Attributes:
            trees {list}: 1d list with RegressionTree objects
            lr {float}: Learning rate
            init_val {float}: Initial value to predict
            fn {function}: A function wrapper for prediction
        """

        self.trees = None
        self.lr = None
        self.init_val = None
        self.fn = lambda x: NotImplemented

    def _get_init_val(self, y):
        """Calculate the initial prediction of y

        Arguments:
            y {list} -- 1D list with int or float

        Returns:
            NotImplemented
        """

        return NotImplemented

    def _match_node(self, row, tree):
        """Find the leaf node that the sample belongs to

        Arguments:
            row {list} -- 1D list with int or float
            tree {RegressionTree}

        Returns:
            regression_tree.Node
        """

        nd = tree.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd

    def _get_leaves(self, tree):
        """Gets all leaf nodes of a regression tree

        Arguments:
            tree {RegressionTree}

        Returns:
            list -- 1D list with regression_tree.Node objects
        """

        nodes = []
        que = [tree.root]
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

    def _divide_regions(self, tree, nodes, X):
        """Divide indexes of the samples into corresponding leaf nodes of the regression tree

        Arguments:
            tree {RegressionTree}
            nodes {list} -- 1D list with regression_tree.Node objects
            X {list} -- 2d list object with int or float

        Returns:
            dict -- e.g. {node1: [1, 3, 5], node2: [2, 4, 6]...}
        """

        regions = {node: [] for node in nodes}
        for i, row in enumerate(X):
            node = self._match_node(row, tree)
            regions[node].append(i)
        return regions

    def _get_score(self, idxs, y_hat, residuals):
        """Calculate the regression tree leaf node value

        Arguments:
            idxs {list} -- 1D list with int

        Returns:
            NotImplemented
        """

        return NotImplemented

    def _update_score(self, tree, X, y_hat, residuals):
        """update the score of regression tree leaf node

        Arguments:
            tree {RegressionTree}
            X {list} -- 2d list with int or float
            y_hat {list} -- 1d list with float
            residuals {list} -- 1d list with float
        """

        nodes = self._get_leaves(tree)

        regions = self._divide_regions(tree, nodes, X)
        for node, idxs in regions.items():
            node.score = self._get_score(idxs, y_hat, residuals)
        tree._get_rules()

    def _get_residuals(self, y, y_hat):
        """Update residuals for each iteration

        Arguments:
            y {list} -- 1d list with int or float
            y_hat {list} -- 1d list with float

        Returns:
            list -- residuals
        """

        return [yi - self.fn(y_hat_i) for yi, y_hat_i in zip(y, y_hat)]

    def fit(self, X, y, n_estimators, lr, max_depth, min_samples_split, subsample=None):
        """Build a gradient boost decision tree.

        Arguments:
            X {list} -- 2d list with int or float
            y {list} -- 1d list object with int or float
            n_estimators {int} -- number of trees
            lr {float} -- Learning rate
            max_depth {int} -- The maximum depth of the tree.
            min_samples_split {int} -- The minimum number of samples required to split an internal node.


        Keyword Arguments:
            subsample {float} -- Subsample rate, without replacement (default: {None})
        """

        # Calculate the initial prediction of y
        self.init_val = self._get_init_val(y)
        # Initialize y_hat
        n = len(y)
        y_hat = [self.init_val] * n
        # Initialize the residuals
        residuals = self._get_residuals(y, y_hat)
        # Train Regression Trees
        self.trees = []
        self.lr = lr
        for _ in range(n_estimators):
            # Sampling with replacement
            idx = range(n)
            if subsample is not None:
                k = int(subsample * n)
                idx = choices(population=idx, k=k)
            X_sub = [X[i] for i in idx]
            residuals_sub = [residuals[i] for i in idx]
            y_hat_sub = [y_hat[i] for i in idx]
            # Train a Regression Tree by sub-sample of X, residuals
            tree = RegressionTree()
            tree.fit(X_sub, residuals_sub, max_depth, min_samples_split)
            # Update scores of tree leaf nodes
            self._update_score(tree, X_sub, y_hat_sub, residuals_sub)
            # Update y_hat
            y_hat = [y_hat_i + lr * res_hat_i for y_hat_i,
                     res_hat_i in zip(y_hat, tree.predict(X))]
            # Update residuals
            residuals = self._get_residuals(y, y_hat)
            self.trees.append(tree)

    def _predict(self, row):
        """Auxiliary function of predict.

        Arguments:
            row {list} -- 1D list with int or float

        Returns:
            int or float -- prediction of yi
        """

        # Sum y_hat with residuals of each tree and then calulate sigmoid value
        return self.fn(self.init_val + sum(self.lr * tree._predict(row) for tree in self.trees))

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            NotImplemented
        """

        return NotImplemented
