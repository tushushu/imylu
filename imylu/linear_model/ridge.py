# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-27 11:25:30
@Last Modified by: tushushu
@Last Modified time: 2018-06-27 11:25:30
"""
from .linear_regression import LinearRegression


class Ridge(LinearRegression):
    """Ridge regression class.

    Attributes:
        bias: b
        weights: W
        alpha: α
    """

    def _get_gradient_delta(self, Xi, yi):
        """Calculate the gradient delta of the partial derivative of MSE
        Loss function:
        L = (y - y_hat) ^ 2 + L2
        L = (y - W * X - b) ^ 2 + α * (W, b) ^ 2

        Get partial derivative of W:
        dL/dW = -2 * (y - W * X - b) * X + 2 * α * W
        dL/dW = -2 * (y - y_hat) * X + 2 * α * W

        Get partial derivative of b:
        dL/db = -2 * (y - W * X - b) + 2 * α * b
        dL/db = -2 * (y - y_hat) + 2 * α * b
        ----------------------------------------------------------------

        Arguments:
            Xi {list} -- 1d list object with int
            yi {float}

        Returns:
            tuple -- Gradient delta of bias and weight
        """

        assert self.alpha is not None, "Parameter alpha is None!"
        y_hat = self._predict(Xi)
        bias_grad_delta = yi - y_hat - self.alpha * self.bias
        weights_grad_delta = [(yi - y_hat) * Xij - self.alpha * wj
                              for Xij, wj in zip(Xi, self.weights)]
        return bias_grad_delta, weights_grad_delta
