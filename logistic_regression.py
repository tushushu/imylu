# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-07-05 16:41:03 
@Last Modified by: tushushu 
@Last Modified time: 2018-07-05 16:41:03 
"""
from linear_regression import LinearRegression
from utils import min_max_scale, load_breast_cancer, run_time, get_acc, train_test_split
from math import exp


class LogisticRegression(LinearRegression):
    """Logistic regression class, z = WX + b, y = 1 / (1 + e**(-z))

    Attributes:
        bias: b
        weights: W
    """

    def _get_gradient_delta(self, Xi, yi):
        """Calculate the gradient delta of the partial derivative of Loss

        Arguments:
            Xi {list} -- 1d list object with int
            yi {int}

        Returns:
            tuple -- Gradient delta of bias and weight
        """

        z = sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias
        y_hat = 1 / (1 + exp(-z))
        bias_grad_delta = yi - y_hat
        weights_grad_delta = [bias_grad_delta * Xij for Xij in Xi]
        return bias_grad_delta, weights_grad_delta

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1d list object with int or float

        Returns:
            float -- prediction of yi
        """

        z = sum(wi * Xij for wi, Xij in zip(self.weights, Xi)) + self.bias
        return 1 / (1 + exp(-z))

    def predict(self, X, threshold=0.5):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Keyword Arguments:
            threshold {float} -- Prediction = 1 when probability >= threshold (default: {0.5})

        Returns:
            list -- 1d list object with float
        """

        return [int(self._predict(Xi) >= threshold) for Xi in X]


def main():
    @run_time
    def batch():
        print("Tesing the accuracy of LogisticRegression(batch)...")
        # Train model
        clf = LogisticRegression()
        clf.fit(X=X_train, y=y_train, lr=0.008, epochs=5000)
        # Model accuracy
        get_acc(clf, X_test, y_test)

    @run_time
    def stochastic():
        print("Tesing the accuracy of LogisticRegression(stochastic)...")
        # Train model
        clf = LogisticRegression()
        clf.fit(X=X_train, y=y_train, lr=0.01, epochs=200,
                method="stochastic", sample_rate=0.5)
        # Model accuracy
        get_acc(clf, X_test, y_test)

    # Load data
    X, y = load_breast_cancer()
    X = min_max_scale(X)
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    batch()
    stochastic()


if __name__ == "__main__":
    main()
