# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-27 11:25:30
@Last Modified by: tushushu
@Last Modified time: 2018-06-27 11:25:30
"""
from random import sample, normalvariate
from utils import load_boston_house_prices, train_test_split, get_r2, run_time, min_max_scale


class LinearRegression(object):
    def __init__(self):
        """Linear regression class, y = WX + b

        Attributes:
            bias: b
            weights: W
        """

        self.bias = None
        self.weights = None

    def _get_gradient_delta(self, Xi, yi):
        """Calculate the gradient delta of the partial derivative of MSE

        Arguments:
            Xi {list} -- 1d list object with int
            yi {float}

        Returns:
            tuple -- Gradient delta of bias and weight
        """

        bias_grad_delta = yi - \
            sum(wi * xij for wi, xij in zip(self.weights, Xi)) - self.bias
        weights_grad_delta = [bias_grad_delta * Xij for Xij in Xi]
        return bias_grad_delta, weights_grad_delta

    def _batch_gradient_descent(self, X, y, lr, epochs):
        """Update the gradient by the whole dataset

        Arguments:
            X {list} -- 2D list with int or float
            y {list} -- 1D list with int or float
            lr {float} -- Learning rate
            epochs {int} -- Number of epochs to update the gradient
        """

        m, n = len(X), len(X[0])
        self.bias = 0
        self.weights = [normalvariate(0, 0.01) for _ in range(n)]

        for _ in range(epochs):
            bias_grad = 0
            weights_grad = [0 for _ in range(n)]

            for i in range(m):
                bias_grad_delta, weights_grad_delta = self._get_gradient_delta(
                    X[i], y[i])
                bias_grad += bias_grad_delta
                weights_grad = [w_grad + w_grad_d for w_grad,
                                w_grad_d in zip(weights_grad, weights_grad_delta)]

            self.bias += lr * bias_grad * 2 / m
            self.weights = [w + lr * w_grad * 2 / m for w,
                            w_grad in zip(self.weights, weights_grad)]

    def _stochastic_gradient_descent(self, X, y, lr, epochs, sample_rate):
        """Update the gradient by the random sample of dataset

        Arguments:
            X {list} -- 2D list with int or float
            y {list} -- 1D list with int or float
            lr {float} -- Learning rate
            epochs {int} -- Number of epochs to update the gradient
            sample_rate {float} -- Between 0 and 1
        """

        m, n = len(X), len(X[0])
        k = int(m * sample_rate)
        self.bias = 0
        self.weights = [normalvariate(0, 0.01) for _ in range(n)]

        for _ in range(epochs):
            for i in sample(range(m), k):
                bias_grad, weights_grad = self._get_gradient_delta(X[i], y[i])
                self.bias += lr * bias_grad
                self.weights = [w + lr * w_grad for w,
                                w_grad in zip(self.weights, weights_grad)]

    def fit(self, X, y, lr, epochs, method="batch", sample_rate=1.0):
        """Train linear regression model

        Arguments:
            X {list} -- 2D list with int or float
            y {list} -- 1D list with int or float
            lr {float} -- Learning rate
            epochs {int} -- Number of epochs to update the gradient

        Keyword Arguments:
            method {str} -- "batch" or "stochastic" (default: {"batch"})
            sample_rate {float} -- Between 0 and 1 (default: {1.0})
        """

        assert method in ("batch", "stochastic")
        # batch gradient descent
        if method == "batch":
            self._batch_gradient_descent(X, y, lr, epochs)
        # stochastic gradient descent
        if method == "stochastic":
            self._stochastic_gradient_descent(X, y, lr, epochs, sample_rate)

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1d list object with int or float

        Returns:
            float -- prediction of yi
        """

        return sum(wi * Xij for wi, Xij in zip(self.weights, Xi)) + self.bias

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2D list with int or float

        Returns:
            list -- prediction of y
        """

        return [self._predict(xi) for xi in X]


def main():
    @run_time
    def batch():
        print("Tesing the accuracy of LinearRegression(batch)...")
        # Train model
        reg = LinearRegression()
        reg.fit(X=X_train, y=y_train, lr=0.02, epochs=5000)
        # Model accuracy
        get_r2(reg, X_test, y_test)

    @run_time
    def stochastic():
        print("Tesing the accuracy of LinearRegression(stochastic)...")
        # Train model
        reg = LinearRegression()
        reg.fit(X=X_train, y=y_train, lr=0.001, epochs=1000,
                method="stochastic", sample_rate=0.5)
        # Model accuracy
        get_r2(reg, X_test, y_test)

    # Load data
    X, y = load_boston_house_prices()
    X = min_max_scale(X)
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    batch()
    stochastic()


if __name__ == "__main__":
    main()
