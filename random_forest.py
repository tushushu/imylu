# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-06-26 14:41:08 
@Last Modified by: tushushu 
@Last Modified time: 2018-06-26 14:41:08 
"""
from random import sample, choice
from decision_tree import DecisionTree


class RandomForest(object):
    def __init__(self):
        """RandomForest, randomly build some DecisionTree instance, 
        and the average score of each DecisionTree

        Attributes:
        trees {list} -- 1d list with DecisionTree objects
        """

        self.trees = None
        self.tree_features = None

    def fit(self, X, y, n_estimators=10, max_depth=3, min_samples_split=2, max_features=None, n_samples=None):
        """Build a RandomForest classifier.

        Arguments:
            X {list} -- 2d list with int or float
            y {list} -- 1d list object with int 0 or 1

        Keyword Arguments:
            n_estimators {int} -- number of trees (default: {5})
            max_depth {int} -- The maximum depth of each tree. (default: {3})
            min_samples_split {int} -- The minimum number of samples required to split an internal node (default: {2})
            n_samples {[type]} -- number of samples (default: {None})
        """

        self.trees = []
        self.tree_features = []
        for _ in range(n_estimators):

            m = len(X[0])
            n = len(y)
            if n_samples:
                idx = sample(range(n), n_samples)
            else:
                idx = range(n)
            if max_features:
                n_features = min(m, max_features)
            else:
                n_features = int(m ** 0.5)

            features = sample(range(m), choice(range(1, n_features)))
            X_sub = [[X[i][j] for j in features] for i in idx]
            y_sub = [y[i] for i in idx]
            clf = DecisionTree()
            clf.fit(X_sub, y_sub, max_depth, min_samples_split)
            self.trees.append(clf)
            self.tree_features.append(features)

    def _predict(self, row):
        """Auxiliary function of predict.

        Arguments:
            row {list} -- 1d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        # Vote
        pos_vote = 0
        for tree, features in zip(self.trees, self.tree_features):
            score = tree._predict_prob([row[i] for i in features])
            if score >= 0.5:
                pos_vote += 1
        neg_vote = len(self.trees) - pos_vote
        if pos_vote > neg_vote:
            return 1
        elif pos_vote < neg_vote:
            return 0
        else:
            return choice([0, 1])

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        return [self._predict(row) for row in X]


if __name__ == "__main__":
    from time import time
    from utils import load_breast_cancer, train_test_split

    start = time()
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)
    # Train RandomForest model
    rf = RandomForest()
    rf.fit(X_train, y_train, n_samples=300, max_depth=3, n_estimators=20)
    # RandomForest Model accuracy
    rf_acc = sum((y_test_hat == y_test for y_test_hat, y_test in zip(
        rf.predict(X_test), y_test))) / len(y_test)
    print("RF accuracy is %.2f%%!" % (rf_acc * 100))
    # Train DecisionTree model
    dt = DecisionTree()
    dt.fit(X_train, y_train, max_depth=4)
    # DecisionTree Model accuracy
    dt_acc = sum((y_test_hat == y_test for y_test_hat, y_test in zip(
        dt.predict(X_test), y_test))) / len(y_test)
    print("DT accuracy is %.2f%%!" % (dt_acc * 100))
    # Show run time, you can try it in Pypy which might be 10x faster.
    print("Total run time is %.2f s" % (time() - start))
