# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-10-29 11:55:57
@Last Modified by:   tushushu
@Last Modified time: 2018-10-29 11:55:57
http://python.jobbole.com/88799/
https://www.jianshu.com/p/997e0ee1e010
"""
from random import normalvariate


class Lasso(object):
    def __init__(self):
        """Lasso regression class.

        Attributes:
            bias: b
            weights: W
            alpha: α
        """

        self.bias = None
        self.weights = None
        self.alpha = None

    def _predict(self, Xi):
        """y = WX + b.

        Arguments:
            Xi {list} -- 1d list object with int or float.

        Returns:
            float -- y
        """

        return sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias

    def _get_coordinate_delta(self, Xi, yi):
        """Calculate the coordinate delta of the partial derivative.

        Arguments:
            Xi {list} -- 1d list object with int.
            yi {float}

        Returns:
            NotImplemented
        """

        return NotImplemented

    def fit(self, X, y, lr, epochs):
        """Update the coordinate by the whole dataset.
        b = b - learning_rate * 1/m * b_grad_i, b_grad_i <- grad
        W = W - learning_rate * 1/m * w_grad_i, w_grad_i <- grad

        Arguments:
            X {list} -- 2D list with int or float.
            y {list} -- 1D list with int or float.
            lr {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the coordinate.
        """

        m, n = len(X), len(X[0])
        self.bias = 0
        self.weights = [normalvariate(0, 0.01) for _ in range(n)]
        # Calculate the gradient of each epoch(iteration)
        for _ in range(epochs):
            bias_grad = 0
            weights_grad = [0 for _ in range(n)]
            # Calculate and sum the gradient delta of each sample
            for i in range(m):
                bias_grad_delta, weights_grad_delta = self._get_gradient_delta(
                    X[i], y[i])
                bias_grad += bias_grad_delta
                weights_grad = [w_grad + w_grad_d for w_grad, w_grad_d
                                in zip(weights_grad, weights_grad_delta)]
            # Update the bias and weight by gradient of current epoch
            self.bias += lr * bias_grad * 2 / m
            self.weights = [w + lr * w_grad * 2 / m for w,
                            w_grad in zip(self.weights, weights_grad)]

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2D list with int or float.

        Returns:
            NotImplemented
        """

        return NotImplemented
