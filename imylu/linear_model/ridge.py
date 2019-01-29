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

    Attributes:
        bias: b
        weights: W
        alpha: α
    """

    def __init__(self):
        LinearRegression.__init__(self)
        self.alpha = None

    def _get_gradient(self, X, y):
        """Calculate the gradient of the partial derivative.

        Arguments:
            X {array} -- 2d array object with int.
            y {float}

        Returns:
            tuple -- Gradient of bias and weight
        """

        y_hat = self.predict(X)
        bias_grad = y - y_hat - self.alpha * self.bias
        try:
            weights_grad = (y - y_hat)[:, None] * X - self.alpha * self.weights
        except IndexError:
            weights_grad = (y - y_hat) * X - self.alpha * self.weights
        return bias_grad, weights_grad

    def fit(self, X, y, lr, epochs, alpha, method="batch", sample_rate=1.0):
        """Train regression model.

        Arguments:
            X {list} -- 2D list with int or float.
            y {list} -- 1D list with int or float.
            lr {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the gradient.
            alpha {float} -- Regularization strength.

        Keyword Arguments:
            method {str} -- "batch" or "stochastic" (default: {"batch"})
            sample_rate {float} -- Between 0 and 1 (default: {1.0})
        """

        self.alpha = alpha
        assert method in ("batch", "stochastic")
        # batch gradient descent
        if method == "batch":
            self._batch_gradient_descent(X, y, lr, epochs)
        # stochastic gradient descent
        if method == "stochastic":
            self._stochastic_gradient_descent(X, y, lr, epochs, sample_rate)
