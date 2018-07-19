# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-07-05 17:37:34 
@Last Modified by: tushushu 
@Last Modified time: 2018-07-05 17:37:34 
"""
from regression_tree import RegressionTree
from copy import copy
from utils import load_boston_house_prices, train_test_split, get_r2, run_time
from random import sample
from math import log, exp


class GradientBoostingRegressor(object):
    def __init__(self):
        """GBDT class for regression.

        Attributes:
            trees {list}: 1d list with RegressionTree objects
            lr {float}: Learning rate
        """

        self.trees = None
        self.lr = None
        self.init_val = None

    def fit(self, X, y, n_estimators, lr, max_depth, min_samples_split, subsample=None):
        """Build a gradient boost decision tree.
        Set MSE as loss function, and c is a constant:
        L = MSE(y, c) = Sum((yi-c) ^ 2) / m, yi <- y

        Get derivative of c:
        dL / dc = Sum(2 * (yi-c)) / m
        dL / dc = 2 * (Sum(yi) / m - Sum(c) / m)
        dL / dc = 2 * (Mean(yi) - c)

        Let derivative of y equals to zero, then we get initial constant value to minimize MSE:
        2 * (Mean(yi) - c) = 0
        c = Mean(yi)
        ----------------------------------------------------------------------------------------

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

        n = len(y)
        self.trees = []
        self.lr = lr
        self.init_val = sum(y) / n
        residual = [yi - self.init_val for yi in y]
        for _ in range(n_estimators):
            # Sampling without replacement
            if subsample is None:
                idx = range(n)
            else:
                k = int(subsample * n)
                idx = sample(range(n), k)
            X_sub = [X[i] for i in idx]
            residual_sub = [residual[i] for i in idx]
            # Train Regression Tree by sub-sample of X, y
            tree = RegressionTree()
            tree.fit(X_sub, residual_sub, max_depth, min_samples_split)
            # Calculate residual
            residual = [a - lr * b for a,
                        b in zip(residual, tree.predict(X))]
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
    reg = GradientBoostingRegressor()
    reg.fit(X=X_train, y=split_train, n_estimators=100,
            lr=0.1, max_depth=2, min_samples_split=2, subsample=0.95)
    # Model accuracy
    get_r2(reg, X_test, split_test)


if __name__ == "__main__":
    main()
