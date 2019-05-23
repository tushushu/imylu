# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-27 11:25:30
@Last Modified by: tushushu
@Last Modified time: 2018-06-27 11:25:30
"""
from numpy import array
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

    def predict(self, data) -> array:
        """Get the prediction of label.

        Arguments:
            data {array} -- Testing data.

        Returns:
            array -- Prediction of label.
        """

        return data.dot(self.weights) + self.bias
