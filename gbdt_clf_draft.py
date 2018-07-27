# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-05 17:37:34
@Last Modified by: tushushu
@Last Modified time: 2018-07-05 17:37:34
"""
from regression_tree import RegressionTree
from copy import copy
from utils import load_boston_house_prices, train_test_split, get_r2, run_time, sigmoid
from random import sample
from math import log, exp


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

        Arguments:
            y {list} -- 1d list object with int or float

        Returns:
            float
        """

        n = len(y)
        y_sum = sum(y)
        return 0.5 * log((y_sum) / (n - y_sum))

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
        # Train Regression Trees
        n = len(y)
        self.trees = []
        self.lr = lr
        for _ in range(n_estimators):
            # Sampling without replacement
            if subsample is None:
                idx = range(n)
            else:
                k = int(subsample * n)
                idx = sample(range(n), k)
            X_sub = [X[i] for i in idx]
            residuals_sub = [residuals[i] for i in idx]
            # Train a Regression Tree by sub-sample of X, residuals
            tree = RegressionTree()
            tree.fit(X_sub, residuals_sub, max_depth, min_samples_split)
            # Calculate residuals
            residuals = [residual - lr * residual_hat for residual,
                         residual_hat in zip(residuals, tree.predict(X))]
            self.trees.append(tree)

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            row {list} -- 1D list with int or float

        Returns:
            int or float -- prediction of yi
        """

        # Sum y_hat with residuals of each tree
        return self.init_val + sum(self.lr * tree._predict(Xi) for tree in self.trees)

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with int or float
        """

        return [self._predict(Xi) for Xi in X]


@run_time
def main():
    print("Tesing the accuracy of GBDT...")
    # Load data
    X, y = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    X_train, X_test, split_train, split_test = train_test_split(
        X, y, random_state=10)
    # Train model
    clf = GradientBoostingClassifier()
    clf.fit(X=X_train, y=split_train, n_estimators=100,
            lr=0.1, max_depth=2, min_samples_split=2, subsample=0.95)
    # Model accuracy
    get_r2(clf, X_test, split_test)


if __name__ == "__main__":
    main()
