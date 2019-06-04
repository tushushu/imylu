# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-05 16:41:03
@Last Modified by: tushushu
@Last Modified time: 2018-07-05 16:41:03
"""
from numpy import ndarray

from .regression_base import RegressionBase
from ..utils.utils import sigmoid


class LogisticRegression(RegressionBase):
    """Logistic regression class.
    Estimation function (Maximize the likelihood):
    z = WX + b
    y = 1 / (1 + e**(-z))

    Likelihood function:
    P(y | X, W, b) = y_hat^y * (1-y_hat)^(1-y)
    L = Product(P(y | X, W, b))

    Take the logarithm of both sides of this equation:
    log(L) = Sum(log(P(y | X, W, b)))
    log(L) = Sum(log(y_hat^y * (1-y_hat)^(1-y)))
    log(L) = Sum(y * log(y_hat) + (1-y) * log(1-y_hat)))

    Get partial derivative of W and b:
    1. dz/dW = X
    2. dy_hat/dz = y_hat * (1-y_hat)
    3. dlog(L)/dy_hat = y * 1/y_hat - (1-y) * 1/(1-y_hat)
    4. dz/db = 1


    According to 1,2,3:
    dlog(L)/dW = dlog(L)/dy_hat * dy_hat/dz * dz/dW
    dlog(L)/dW = (y - y_hat) * X

    According to 2,3,4:
    dlog(L)/db = dlog(L)/dy_hat * dy_hat/dz * dz/db
    dlog(L)/db = y - y_hat
    ----------------------------------------------------------------

    Attributes:
        bias: b
        weights: W
    """

    def predict_prob(self, data: ndarray):
        """Get the probability of label.

        Arguments:
            data {ndarray} -- Testing data.

        Returns:
            ndarray -- Probabilities of label.
        """

        return sigmoid(data.dot(self.weights) + self.bias)

    def predict(self, data: ndarray, threshold=0.5):  # pylint: disable=arguments-differ
        """Get the prediction of label.

        Arguments:
            data {ndarray} -- Testing data.

        Keyword Arguments:
            threshold {float} -- (default: {0.5})

        Returns:
            ndarray -- Prediction of label.
        """

        prob = self.predict_prob(data)
        return (prob >= threshold).astype(int)
