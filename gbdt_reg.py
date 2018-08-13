# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-07-05 17:37:34 
@Last Modified by: tushushu 
@Last Modified time: 2018-07-05 17:37:34 
"""
from regression_tree import RegressionTree
from utils import load_boston_house_prices, train_test_split, get_r2, run_time
from random import choices
from gbdt_base import GradientBoostingBase


class GradientBoostingRegressor(GradientBoostingBase):
    def __init__(self):
        super(GradientBoostingRegressor, self).__init__()
        self.fn = lambda x: x

    def _get_init_val(self, y):
        """Calculate the initial prediction of y
        Set MSE as loss function, yi <- y, and c is a constant:
        L = MSE(y, c) = Sum((yi-c) ^ 2) / m

        Get derivative of c:
        dL / dc = Sum(2 * (yi-c)) / m
        dL / dc = 2 * (Sum(yi) / m - Sum(c) / m)
        dL / dc = 2 * (Mean(yi) - c)

        Let derivative equals to zero, then we get initial constant value to minimize MSE:
        2 * (Mean(yi) - c) = 0
        c = Mean(yi)
        ----------------------------------------------------------------------------------------

        Arguments:
            y {list} -- 1d list object with int or float

        Returns:
            float
        """

        return sum(y) / len(y)

    def _update_score(self, tree, X, y_hat, residuals):
        """[summary]

        Arguments:
            tree {RegressionTree}
            X {list} -- 2d list with int or float
        """

        pass

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with int or float
        """

        return [self._predict(row) for row in X]


@run_time
def main():
    print("Tesing the accuracy of GBDT regressor...")
    # Load data
    X, y = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10)
    # Train model
    reg = GradientBoostingRegressor()
    reg.fit(X=X_train, y=y_train, n_estimators=4,
            lr=0.5, max_depth=2, min_samples_split=2)
    # Model accuracy
    get_r2(reg, X_test, y_test)


if __name__ == "__main__":
    main()
