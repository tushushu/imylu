# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-27 11:25:30
@Last Modified by: tushushu
@Last Modified time: 2018-06-27 11:25:30
"""
import numpy as np
from numpy.random import choice
from numpy import array
from ..utils.utils import arr2str


class RegressionBase(object):
    def __init__(self):
        """Regression base class.

        Attributes:
            bias: b
            weights: W
        """

        self.bias = None
        self.weights = None

    def __str__(self):
        weights = arr2str(self.weights, 2)
        return "Weighs: %s\nBias: %.2f\n" % (weights, self.bias)

    def _get_gradient(self, X: array, y: array):
        """Calculate the gradient of the partial derivative.

        Arguments:
            X {array} -- 2d array object with int.
            y {float}

        Returns:
            tuple -- Gradient of bias and weight
        """

        # Use predict_prob method if this is a classifier.
        if hasattr(self, "predict_prob"):
            y_hat = self.predict_prob(X)
        else:
            y_hat = self.predict(X)

        # Calculate the gradient according to the dimention of X, y.
        grad_bias = y - y_hat
        if X.ndim is 1:
            grad_weights = grad_bias * X
        elif X.ndim is 2:
            grad_weights = grad_bias[:, None] * X
            grad_weights = grad_weights.mean(axis=0)
            grad_bias = grad_bias.mean()
        else:
            raise ValueError("Dimension of X has to be 1 or 2!")
        return grad_bias, grad_weights

    def _batch_gradient_descent(self, X, y, lr, epochs):
        """Update the gradient by the whole dataset.
        b = b - learning_rate * 1/m * b_grad_i, b_grad_i <- grad
        W = W - learning_rate * 1/m * w_grad_i, w_grad_i <- grad

        Arguments:
            X {array} -- 2D array with int or float.
            y {array} -- 1D array with int or float.
            lr {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the gradient.
        """

        # Initialize the bias and weights.
        _, n = X.shape
        self.bias = 0
        self.weights = np.random.normal(size=n)

        for i in range(epochs):
            # Calculate and sum the gradient delta of each sample
            grad_bias, grad_weights = self._get_gradient(X, y)

            # Show the gradient of each epoch.
            grad = (grad_bias + grad_weights.mean()) / 2
            print("Epochs %d gradient %.3f" % (i + 1, grad), flush=True)

            # Update the bias and weight by gradient of current epoch
            self.bias += lr * grad_bias
            self.weights += lr * grad_weights

    def _stochastic_gradient_descent(self, X, y, lr, epochs, sample_rate):
        """Update the gradient by the random sample of dataset.
        b = b - learning_rate * b_sample_grad_i, b_sample_grad_i <- sample_grad
        W = W - learning_rate * w_sample_grad_i, w_sample_grad_i <- sample_grad

        Arguments:
            X {array} -- 2D array with int or float.
            y {array} -- 1D array with int or float.
            lr {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the gradient.
            sample_rate {float} -- Between 0 and 1.
        """

        # Initialize the bias and weights.
        m, n = X.shape
        self.bias = 0
        self.weights = np.random.normal(size=n)

        n_sample = int(m * sample_rate)
        for i in range(epochs):
            for idx in choice(range(m), n_sample, replace=False):
                # Calculate the gradient delta of each sample
                grad_bias, grad_weights = self._get_gradient(X[idx], y[idx])

                # Update the bias and weight by gradient of current sample
                self.bias += lr * grad_bias
                self.weights += lr * grad_weights

            # Show the gradient of each epoch.
            grad_bias, grad_weights = self._get_gradient(X, y)
            grad = (grad_bias + grad_weights.mean()) / 2
            print("Epochs %d gradient %.3f" % (i + 1, grad), flush=True)

    def fit(self, X: array, y: array, lr: float, epochs: int,
            method: str = "batch", sample_rate: float = 1.0):
        """Train regression model.

        Arguments:
            X {array} -- 2D array with int or float.
            y {array} -- 1D array with int or float.
            lr {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the gradient.

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

    def predict(self, X: array):
        """Get the prediction of y.

        Arguments:
            X {array} -- 2D array with int or float.

        Returns:
            NotImplemented
        """

        return NotImplemented
