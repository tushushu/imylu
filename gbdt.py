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


class GBDT(object):
    def __init__(self):
        self.trees = None

    def fit(self, X, y, n_estimators, max_depth, min_samples_split):
        self.trees = []
        residual = copy(y)
        for _ in range(n_estimators):
            tree = RegressionTree()
            tree.fit(X, residual, max_depth, min_samples_split)
            residual = [a - b for a, b in zip(residual, tree.predict(X))]
            self.trees.append(tree)

    def _predict(self, Xi):
        return sum(tree._predict(Xi) for tree in self.trees)

    def predict(self, X):
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
    reg = GBDT()
    reg.fit(X=X_train, y=split_train, n_estimators=10,
            max_depth=2, min_samples_split=2)
    # Model accuracy
    get_r2(reg, X_test, split_test)


if __name__ == "__main__":
    main()
