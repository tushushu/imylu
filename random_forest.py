# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-06-26 14:41:08 
@Last Modified by: tushushu 
@Last Modified time: 2018-06-26 14:41:08 
"""
from decision_tree import DecisionTree


class RandomForest(object):
    def __init__(self):
        """RandomForest, randomly build some DecisionTree instance, 
        and the average score of each DecisionTree

        Attributes:
        trees {list} -- 1d list with DecisionTree objects
        """

        self.trees = None

    def fit(self, X, y, n_estimators=10, max_depth=3, min_samples_split=2):
        return None

    def _predict(self):
        None

    def predict(self):
        None


if __name__ == "__main__":
    from time import time
    from utils import load_breast_cancer, train_test_split

    start = time()
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
    # Train model
    clf = RandomForest()
    clf.fit(X_train, y_train)
    # Show rules
    clf.print_rules()
    # Model accuracy
    acc = sum((y_test_hat == y_test for y_test_hat, y_test in zip(
        clf.predict(X_test), y_test))) / len(y_test)
    print("Test accuracy is %.2f%%!" % (acc * 100))
    # Show run time, you can try it in Pypy which might be 10x faster.
    print("Total run time is %.2f s" % (time() - start))
