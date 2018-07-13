# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-07-06 21:13:34 
@Last Modified by: tushushu 
@Last Modified time: 2018-07-06 21:13:34 
"""
from utils import load_breast_cancer, train_test_split, get_acc, run_time
from math import pi, exp, sqrt


class GaussianNB(object):
    def __init__(self):
        self.vals = None

    def _get_cond_prob(self, feature, xij):
        avg, var = self.vals[feature]
        cond_prob = 1 / sqrt(2*pi*var) * exp(-(xij-avg)**2 / (2*var))
        return cond_prob

    def fit(self, X, y):
        self.vals = []
        m = len(X[0])
        n = len(X)
        feature_sum = feature_sqr_sum = 0
        for j in range(m):
            for i in range(n):
                feature_sum += X[i][j]
                feature_sqr_sum += X[i][j] ** 2
            feature_avg = feature_sum / n
            feature_var = feature_sqr_sum / n - feature_avg ** 2
            self.vals.append([feature_avg, feature_var])

    def _predict_prob(self, row):
        """Auxiliary function of predict_prob.

        Arguments:
            row {list} -- 1D list with int or float

        Returns:
            float
        """

        raise NotImplementedError

    def predict_prob(self, X):
        """Get the probability that y is positive.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        return [self._predict_prob(row) for row in X]

    def predict(self, X, threshold=0.5):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Keyword Arguments:
            threshold {float} -- Prediction = 1 when probability >= threshold (default: {0.5})

        Returns:
            list -- 1d list object with float
        """

        return [int(y >= threshold) for y in self.predict_prob(X)]


@run_time
def main():
    print("Tesing the accuracy of NaiveBayes...")
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    # Model accuracy
    get_acc(clf, X_test, y_test)


if __name__ == "__main__":
    main()
