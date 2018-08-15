# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-05 17:37:34
@Last Modified by: tushushu
@Last Modified time: 2018-07-05 17:37:34
"""
from math import exp, log
from random import choices

from regression_tree import RegressionTree
from utils import (get_acc, load_breast_cancer,
                   run_time, sigmoid, train_test_split)

from gbdt_base import GradientBoostingBase


class GradientBoostingClassifier(GradientBoostingBase):
    def __init__(self):
        super(GradientBoostingClassifier, self).__init__()
        self.fn = lambda x: sigmoid(x)

    def _get_init_val(self, y):
        """Calculate the initial prediction of y
        Estimation function (Maximize the likelihood):
        z = fm(xi)
        p = 1 / (1 + e**(-z))

        Likelihood function, yi <- y, and p is a constant:
        Likelihood = Product(p^yi * (1-p)^(1-yi))

        Loss function:
        L = Sum(yi * Logp + (1-yi) * Log(1-p))

        Get derivative of p:
        dL / dp = Sum(yi/p - (1-yi)/(1-p))

        dp / dz = p * (1 - p)

        dL / dz = dL / dp * dp / dz
        dL / dz = Sum(yi * (1-p) - (1-yi)* p)
        dL / dz = Sum(yi) - Sum(1) * p

        Let derivative equals to zero, then we get initial constant value to maximize Likelihood:
        p = Mean(yi)
        1 / (1 + e**(-z)) = Mean(yi)
        z = Log(Sum(yi) / Sum(1-yi))
        ----------------------------------------------------------------------------------------

        Arguments:
            y {list} -- 1d list object with int or float

        Returns:
            float
        """

        n = len(y)
        y_sum = sum(y)
        return log((y_sum) / (n - y_sum))

    def _get_score(self, idxs, y_hat, residuals):
        """Calculate the regression tree leaf node value
        Estimation function (Maximize the likelihood):
        z = Fm(xi) = Fm-1(xi) + fm(xi)
        p = 1 / (1 + e**(-z))

        Likelihood function, yi <- y, and p is a constant:
        Likelihood = Product(p^yi * (1-p)^(1-yi))

        Loss Function:
        Loss(yi, Fm(xi)) = Sum(yi * Logp + (1-y) * Log(1-p))

        Taylor 1st:
        f(x + x_delta) = f(x) + f'(x) * x_delta
        f(x) = g'(x)
        g'(x + x_delta) = g'(x) + g"(x) * x_delta


        1st derivative:
        Loss'(yi, Fm(xi)) = Sum(yi - Sigmoid(Fm(xi)))

        2nd derivative:
        Loss"(yi, Fm(xi)) = Sum((Fm(xi) - 1) * Fm(xi))

        So,
        Loss'(yi, Fm(xi)) = Loss'(yi, Fm-1(xi)) + Loss"(yi, Fm-1(xi)) *  fm(xi) = 0
        fm(xi) = - Loss'(yi, Fm-1(xi)) / Loss"(yi, Fm-1(xi))
        fm(xi) = Sum(yi - Sigmoid(Fm-1(xi))) / Sum((1 - Fm-1(xi)) * Fm-1(xi))
        fm(xi) = Sum(residual_i) / Sum((1 - Fm-1(xi)) * Fm-1(xi))
        ----------------------------------------------------------------------------------------

        Arguments:
            idxs{list} -- 1d list object with int
            y_hat {list} -- 1d list object with int or float
            residuals {list} -- 1d list object with int or float

        Returns:
            float
        """

        numerator = denominator = 0
        for idx in idxs:
            numerator += residuals[idx]
            denominator += y_hat[idx] * (1 - y_hat[idx])
        return numerator / denominator

    def predict(self, X, threshold=0.5):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Keyword Arguments:
            threshold {float} -- Prediction = 1 when probability >= threshold (default: {0.5})

        Returns:
            list -- 1d list object with float
        """

        return [int(self._predict(row) >= threshold) for row in X]


@run_time
def main():
    print("Tesing the accuracy of GBDT Classifier...")
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)
    # Train model
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train, n_estimators=2,
            lr=0.8, max_depth=3, min_samples_split=2)
    # Model accuracy
    get_acc(clf, X_test, y_test)


if __name__ == "__main__":
    main()
