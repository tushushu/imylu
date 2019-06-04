# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-06 21:13:34
@Last Modified by: tushushu
@Last Modified time: 2019-05-02 16:13:34
"""

from collections import Counter

import numpy as np
from numpy import ndarray, exp, pi, sqrt


class GaussianNB:
    """GaussianNB class support multiple classification.

    Attributes:
        prior: Prior probability.
        avgs: Means of training data. e.g. [[0.5, 0.6], [0.2, 0.1]]
        vars: Variances of training data.
        n_class: number of classes
    """

    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None

    @staticmethod
    def _get_prior(label: ndarray) -> ndarray:
        """Calculate prior probability.

        Arguments:
            label {ndarray} -- Target values.

        Returns:
            array
        """

        cnt = Counter(label)
        prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
        return prior

    def _get_avgs(self, data: ndarray, label: ndarray) -> ndarray:
        """Calculate means of training data.

        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.

        Returns:
            array
        """
        return np.array([data[label == i].mean(axis=0)
                         for i in range(self.n_class)])

    def _get_vars(self, data: ndarray, label: ndarray) -> ndarray:
        """Calculate variances of training data.

        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.

        Returns:
            array
        """
        return np.array([data[label == i].var(axis=0)
                         for i in range(self.n_class)])

    def _get_posterior(self, row: ndarray) -> ndarray:
        """Calculate posterior probability

        Arguments:
            row {ndarray} -- Sample of training data.

        Returns:
            array
        """

        return (1 / sqrt(2 * pi * self.vars) * exp(
            -(row - self.avgs)**2 / (2 * self.vars))).prod(axis=1)

    def fit(self, data: ndarray, label: ndarray):
        """Build a Gauss naive bayes classifier.

        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.
        """

        # Calculate prior probability.
        self.prior = self._get_prior(label)
        # Count number of classes.
        self.n_class = len(self.prior)
        # Calculate the mean.
        self.avgs = self._get_avgs(data, label)
        # Calculate the variance.
        self.vars = self._get_vars(data, label)

    def predict_prob(self, data: ndarray) -> ndarray:
        """Get the probability of label.

        Arguments:
            data {ndarray} -- Testing data.

        Returns:
            ndarray -- Probabilities of label.
            e.g. [[0.02, 0.03, 0.02], [0.02, 0.03, 0.02]]
        """

        # Caculate the joint probabilities of each feature and each class.
        likelihood = np.apply_along_axis(self._get_posterior, axis=1, arr=data)
        probs = self.prior * likelihood
        # Scale the probabilities
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    def predict(self, data: ndarray) -> ndarray:
        """Get the prediction of label.

        Arguments:
            data {ndarray} -- Testing data.

        Returns:
            ndarray -- Prediction of label.
        """

        # Choose the class which has the maximum probability
        return self.predict_prob(data).argmax(axis=1)
