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
    """Regression base class.

    Attributes:
        bias: b
        weights: W
    """

    def __init__(self):

        self.bias = None
        self.weights = None

    def __str__(self):
        weights = arr2str(self.weights, 2)
        return "Weights: %s\nBias: %.2f\n" % (weights, self.bias)

    def _get_gradient(self, data: array, score: array):
        """Calculate the gradient of the partial derivative.

        Arguments:
            data {array} -- Training data.
            score {array} -- Target values.

        Returns:
            tuple -- Gradient of bias and weight
        """

        # Use predict_prob method if this is a classifier.
        if hasattr(self, "predict_prob"):
            y_hat = self.predict_prob(data)
        else:
            y_hat = self.predict(data)

        # Calculate the gradient according to the dimention of data, score.
        grad_bias = score - y_hat
        if data.ndim == 1:
            grad_weights = grad_bias * data
        elif data.ndim == 2:
            grad_weights = grad_bias[:, None] * data
            grad_weights = grad_weights.mean(axis=0)
            grad_bias = grad_bias.mean()
        else:
            raise ValueError("Dimension of data has to be 1 or 2!")

        return grad_bias, grad_weights

    def _batch_gradient_descent(self, data: array, score: array, learning_rate: float, epochs: int):
        """Update the gradient by the whole dataset.
        b = b - learning_rate * 1/m * b_grad_i, b_grad_i <- grad
        W = W - learning_rate * 1/m * w_grad_i, w_grad_i <- grad

        Arguments:
            data {array} -- Training data.
            score {array} -- Target values.
            learning_rate {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the gradient.
        """

        # Initialize the bias and weights.
        _, n_cols = data.shape
        self.bias = 0
        self.weights = np.random.normal(size=n_cols)

        for i in range(epochs):
            # Calculate and sum the gradient delta of each sample.
            grad_bias, grad_weights = self._get_gradient(data, score)

            # Show the gradient of each epoch.
            grad = (grad_bias + grad_weights.mean()) / 2
            print("Epochs %d gradient %.3f" % (i + 1, grad), flush=True)

            # Update the bias and weight by gradient of current epoch
            self.bias += learning_rate * grad_bias
            self.weights += learning_rate * grad_weights

    def _stochastic_gradient_descent(self, data: array, score: array, learning_rate: float,
                                     epochs: int, sample_rate: float):
        """Update the gradient by the random sample of dataset.
        b = b - learning_rate * b_sample_grad_i, b_sample_grad_i <- sample_grad
        W = W - learning_rate * w_sample_grad_i, w_sample_grad_i <- sample_grad

        Arguments:
            data {array} -- Training data.
            score {array} -- Target values.
            learning_rate {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the gradient.
            sample_rate {float} -- Between 0 and 1.
        """

        # Initialize the bias and weights.
        n_rows, n_cols = data.shape
        self.bias = 0
        self.weights = np.random.normal(size=n_cols)

        n_sample = int(n_rows * sample_rate)
        for i in range(epochs):
            for idx in choice(range(n_rows), n_sample, replace=False):
                # Calculate the gradient delta of each sample
                grad_bias, grad_weights = self._get_gradient(
                    data[idx], score[idx])

                # Update the bias and weight by gradient of current sample
                self.bias += learning_rate * grad_bias
                self.weights += learning_rate * grad_weights

            # Show the gradient of each epoch.
            grad_bias, grad_weights = self._get_gradient(data, score)
            grad = (grad_bias + grad_weights.mean()) / 2
            print("Epochs %d gradient %.3f" % (i + 1, grad), flush=True)

    def fit(self, data: array, score: array, learning_rate: float, epochs: int,
            method: str = "batch", sample_rate: float = 1.0):
        """Train regression model.

        Arguments:
            data {array} -- Training data.
            score {array} -- Target values.
            learning_rate {float} -- Learning rate.
            epochs {int} -- Number of epochs to update the gradient.

        Keyword Arguments:
            method {str} -- "batch" or "stochastic" (default: {"batch"})
            sample_rate {float} -- Between 0 and 1 (default: {1.0})
        """

        assert method in ("batch", "stochastic")

        # Batch gradient descent.
        if method == "batch":
            self._batch_gradient_descent(data, score, learning_rate, epochs)

        # Stochastic gradient descent.
        if method == "stochastic":
            self._stochastic_gradient_descent(
                data, score, learning_rate, epochs, sample_rate)

    def predict_prob(self, data: array):
        """Get the probability of label.

        Arguments:
            data {array} -- Training data.

        Returns:
            NotImplemented
        """

        return NotImplemented

    def predict(self, data: array):
        """Get the prediction of score.

        Arguments:
            data {array} -- Training data.

        Returns:
            NotImplemented
        """

        return NotImplemented
