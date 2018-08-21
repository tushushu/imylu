# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:32:24
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 17:32:24
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.utils import load_breast_cancer, train_test_split, get_acc, run_time
from imylu.tree.decision_tree import DecisionTree


@run_time
def main():
    print("Tesing the accuracy of DecisionTree...")
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    clf = DecisionTree()
    clf.fit(X_train, y_train, max_depth=4)
    # Show rules
    clf.rules
    # Model accuracy
    get_acc(clf, X_test, y_test)


if __name__ == "__main__":
    main()
