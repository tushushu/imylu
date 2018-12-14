# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-05 16:41:03
@Last Modified by: tushushu
@Last Modified time: 2018-07-05 16:41:03
"""
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

    def __init__(self):
        RegressionBase.__init__(self)

    def _predict(self, Xi):
        """y = sigmoid(WX + b).

        Arguments:
            Xi {list} -- 1d list object with int or float.

        Returns:
            float -- y
        """
        z = sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Keyword Arguments:
            threshold {float} -- Prediction = 1 when probability >= threshold
            (default: {0.5})

        Returns:
            list -- 1d list object with float
        """

        return [int(self._predict(Xi) >= threshold) for Xi in X]
