# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-27 11:25:30
@Last Modified by: tushushu
@Last Modified time: 2018-06-27 11:25:30
"""
from random import sample, normalvariate


class LinearRegression(object):
    def __init__(self):
        """[summary]

        Arguments:
            object {[type]} -- [description]
        """

        self.bias = None
        self.weights = None

    def get_gradient_delta(self, yi, xi):
        """[summary]

        Arguments:
            yi {[type]} -- [description]
            xi {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        bias_grad_delta = yi - \
            sum(wi * xij for wi, xij in zip(self.weights, xi)) - self.bias
        weights_grad_delta = [d * xi for d in bias_grad_delta]
        return bias_grad_delta, weights_grad_delta

    def batch_gradient_descent(self, x, y, lr, epochs):
        """[summary]

        Arguments:
            x {[type]} -- [description]
            y {[type]} -- [description]
            lr {[type]} -- [description]
            epochs {[type]} -- [description]
        """

        m, n = x.shape
        self.weights = [normalvariate(0, 0.01) for _ in range(n)]
        self.bias = 0

        for _ in range(epochs):
            bias_grad = 0
            weights_grad = [0 for _ in range(n)]

            for i in range(m):
                bias_grad_delta, weights_grad_delta = self.get_gradient_delta(
                    x[i], y[i])
                bias_grad += bias_grad_delta
                weights_grad += weights_grad_delta

            self.bias += lr * bias_grad * 2 / m
            self.weights += lr * weights_grad * 2 / m

    def stochastic_gradient_descent(self, x, y, lr, epochs, sample_rate):
        """[summary]

        Arguments:
            x {[type]} -- [description]
            y {[type]} -- [description]
            lr {[type]} -- [description]
            epochs {[type]} -- [description]
            sample_rate {[type]} -- [description]
        """

        m, n = x.shape
        k = int(m * sample_rate)
        self.weights = [normalvariate(0, 0.01) for _ in range(n)]
        self.bias = 0

        for _ in range(epochs):
            sample(range(m), k)
            for i in sample(range(m), k):
                bias_grad, weights_grad = self.get_gradient_delta(x[i], y[i])

                self.bias += lr * bias_grad
                self.weights += lr * weights_grad

    def fit(self, x, y, lr, epochs, method="batch", sample_rate=1):
        """[summary]

        Arguments:
            x {[type]} -- [description]
            y {[type]} -- [description]
            lr {[type]} -- [description]
            epochs {[type]} -- [description]

        Keyword Arguments:
            method {str} -- [description] (default: {"batch"})
            sample_rate {int} -- [description] (default: {1})
        """

        assert method in ("batch", "stochastic")
        # batch gradient descent
        if method == "batch":
            self.batch_gradient_descent(x, y, lr, epochs)
        # stochastic gradient descent
        if method == "stochastic":
            self.stochastic_gradient_descent(x, y, lr, epochs, sample_rate)

    def _predict(self, xi):
        """[summary]

        Arguments:
            xi {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        """

        return sum(wi * xij for wi, xij in zip(self.weights, xi)) + self.bias

    def predict(self, X):
        """[summary]

        Arguments:
            X {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        """

        return [self._predict(xi) for xi in X]


if __name__ == "__main__":
    reg = LinearRegression()
