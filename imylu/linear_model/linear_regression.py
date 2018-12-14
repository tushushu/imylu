# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-27 11:25:30
@Last Modified by: tushushu
@Last Modified time: 2018-06-27 11:25:30
"""
from .regression_base import RegressionBase


class LinearRegression(RegressionBase):
    """Linear regression class.
    Loss function:
    L = (y - y_hat) ^ 2
    L = (y - W * X - b) ^ 2

    Get partial derivative of W:
    dL/dW = -2 * (y - W * X - b) * X
    dL/dW = -2 * (y - y_hat) * X

    Get partial derivative of b:
    dL/db = -2 * (y - W * X - b)
    dL/db = -2 * (y - y_hat)
    ----------------------------------------------------------------

    Attributes:
        bias: b
        weights: W
    """

    def __init__(self):
        RegressionBase.__init__(self)

    def _predict(self, Xi):
        """y = WX + b.

        Arguments:
            Xi {list} -- 1d list object with int or float.

        Returns:
            float -- y
        """

        return sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2D list with int or float

        Returns:
            list -- prediction of y
        """

        return [self._predict(xi) for xi in X]
