# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-05 17:37:34
@Last Modified by: tushushu
@Last Modified time: 2018-07-05 17:37:34
"""
from math import exp, log
from random import choices

from regression_tree import RegressionTree
from utils import (get_acc, load_breast_cancer,
                   run_time, sigmoid, train_test_split)


class GradientBoostingClassifier(object):
    def __init__(self):
        """GBDT class for binary classification problem.

        Attributes:
            trees {list}: 1d list with RegressionTree objects
            lr {float}: Learning rate
        """

        self.trees = None
        self.lr = None
        self.init_val = None

    def _get_init_val(self, y):
        """Calculate the initial prediction of y
        Estimation function (Maximize the likelihood):
        z = Fm(xi)
        p = 1 / (1 + e**(-z))

        Likelihood function, yi <- y, and p is a constant:
        Likelihood = Product(p^yi * (1-p)^(1-yi))

        Loss function:
        L = Sum(yi * Logp + (1-y) * Log(1-p))

        Get derivative of p:
        dL / dp = Sum(yi/p - (1-yi)/(1-p))
        dp / dz = p * (1 - p)
        dL / dz = dL / dp * dp / dz
        dL / dz = Sum(yi * (1 - p) - (1-yi)* p)
        dL / dz = Sum(yi) - Sum(1) * p

        Let derivative equals to zero, then we get initial constant value to maximize Likelihood:
        p = Mean(yi)
        1 / (1 + e**(-z)) = Mean(yi)
        z = Log(Sum(yi) / Sum(1-yi))
        ----------------------------------------------------------------------------------------

        Arguments:
            y {list} -- 1d list object with int or float

        Returns:
            float
        """

        n = len(y)
        y_sum = sum(y)
        return log((y_sum) / (n - y_sum))

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

    def _get_nodes(self, tree):
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

    def _get_score(self, y_hat, residuals, idxs):
        """Calculate the regression tree leaf node value

        Arguments:
            y_hat {list} -- 1d list object with int or float
            residuals {list} -- 1d list object with int or float
            idxs{list} -- 1d list object with int

        Returns:
            float
        """

        numerator = denominator = 0
        for idx in idxs:
            numerator += residuals[idx]
            denominator += y_hat[idx] * (1 - y_hat[idx])
        return numerator / denominator

    def _update_score(self, tree, X, y_hat, residuals):
        """[summary]

        Arguments:
            tree {RegressionTree}
            X {list} -- 2d list with int or float
            y {list} -- 1d list object with int or float
            residuals {list} -- 1d list object with int or float
        """

        nodes = self._get_nodes(tree)

        regions = self._divide_regions(tree, nodes, X)
        for node, idxs in regions.items():
            node.score = self._get_score(y_hat, residuals, idxs)
        tree._get_rules()

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
        # Initialize the residuals
        residuals = [yi - sigmoid(self.init_val) for yi in y]
        # Initialize y_hat
        n = len(y)
        y_hat = [sigmoid(self.init_val)] * n
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
            # Calculate residuals
            residuals = [residual - lr * sigmoid(residual_hat) for residual,
                         residual_hat in zip(residuals, tree.predict(X))]
            self.trees.append(tree)

    def _predict_prob(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            row {list} -- 1D list with int or float

        Returns:
            int or float -- prediction of yi
        """

        # Sum y_hat with residuals of each tree and then calulate sigmoid value
        return sigmoid(self.init_val + sum(self.lr * tree._predict(Xi) for tree in self.trees))

    def predict_prob(self, X):
        """Get the probability that y is positive.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        return [self._predict_prob(row) for row in X]

    def predict(self, X, threshold=0.5):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Keyword Arguments:
            threshold {float} -- Prediction = 1 when probability >= threshold (default: {0.5})

        Returns:
            list -- 1d list object with float
        """

        return [int(y >= threshold) for y in self.predict_prob(X)]


@run_time
def main():
    print("Tesing the accuracy of GBDT Classifier...")
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train, n_estimators=2,
            lr=0.8, max_depth=3, min_samples_split=2)
    # Model accuracy
    get_acc(clf, X_test, y_test)


if __name__ == "__main__":
    main()
