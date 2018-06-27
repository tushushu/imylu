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

    def get_delta_gradient(self, yi, xi):
        """[summary]

        Arguments:
            yi {[type]} -- [description]
            xi {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        delta_bias_grad = yi - \
            sum(wi * xij for wi, xij in zip(self.weights, xi)) - self.bias
        delta_weights_grad = [d * xi for d in delta_bias_grad]
        return delta_bias_grad, delta_weights_grad

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
            bias_gradient = 0
            weights_gradient = [0 for _ in range(n)]

            for i in range(m):
                delta_bias_gradient, delta_weights_gradient = self.get_delta_gradient(
                    x[i], y[i])
                bias_gradient += delta_bias_gradient
                weights_gradient += delta_weights_gradient

            self.bias += lr * bias_gradient * 2 / m
            self.weights += lr * weights_gradient * 2 / m

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
                bias_gradient, weights_gradient = self.get_delta_gradient(
                    x[i], y[i])

                self.bias += lr * bias_gradient
                self.weights += lr * weights_gradient

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

    def _predict(self, row):
        """[summary]

        Arguments:
            row {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        """

        raise NotImplementedError

    def predict(self, X):
        """[summary]

        Arguments:
            X {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        """

        raise NotImplementedError


if __name__ == "__main__":
    reg = LinearRegression()
