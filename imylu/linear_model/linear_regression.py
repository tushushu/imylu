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

    Attributes:
        bias: b
        weights: W
    """

    def _get_gradient_delta(self, Xi, yi):
        """Calculate the gradient delta of the partial derivative of MSE
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

        Arguments:
            Xi {list} -- 1d list object with int
            yi {float}

        Returns:
            tuple -- Gradient delta of bias and weight
        """

        y_hat = self._predict(Xi)
        bias_grad_delta = yi - y_hat
        weights_grad_delta = [bias_grad_delta * Xij for Xij in Xi]
        return bias_grad_delta, weights_grad_delta

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2D list with int or float

        Returns:
            list -- prediction of y
        """

        return [self._predict(xi) for xi in X]
