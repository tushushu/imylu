# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-27 11:25:30
@Last Modified by: tushushu
@Last Modified time: 2018-06-27 11:25:30
"""
from numpy import array

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
        super(Ridge, self).__init__()
        self.alpha = None

    def _get_gradient(self, data, label):
        """Calculate the gradient of the partial derivative.

        Arguments:
            data {array} -- Training data.
            label {array} -- Target values.

        Returns:
            tuple -- Gradient of bias and weight
        """

        grad_bias, grad_weights = LinearRegression._get_gradient(
            self, data, label)
        grad_bias -= self.alpha * self.bias
        grad_weights -= self.alpha * self.weights

        return grad_bias, grad_weights

    def fit(self, data: array, label: array, learning_rate: float, epochs: int,
            alpha: float, method="batch", sample_rate=1.0, random_state=None):
        """Train regression model.

        Arguments:
            data {array} -- Training data.
            label {array} -- Target values.
            learning_rate {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the gradient.
            alpha {float} -- Regularization strength.

        Keyword Arguments:
            method {str} -- "batch" or "stochastic" (default: {"batch"})
            sample_rate {float} -- Between 0 and 1 (default: {1.0})
            random_state {int} -- The seed used by the random number generator. (default: {None})
        """

        self.alpha = alpha
        LinearRegression.fit(self, data=data, label=label, learning_rate=learning_rate,
                             epochs=epochs, method=method, sample_rate=sample_rate,
                             random_state=random_state)
