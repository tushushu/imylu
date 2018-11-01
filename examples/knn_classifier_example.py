# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-09-14 14:45:50
@Last Modified by:   tushushu
@Last Modified time: 2018-09-14 14:45:50
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.utils import get_acc, load_breast_cancer, run_time, \
    train_test_split, min_max_scale
from imylu.neighbors.knn_classifier import KNeighborsClassifier


@run_time
def main():
    print("Tesing the accuracy of KNN classifier...")
    # Load data
    X, y = load_breast_cancer()
    X = min_max_scale(X)
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)
    # Train model
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train, k_neighbors=21)
    # Model accuracy
    get_acc(clf, X_test, y_test)


if __name__ == "__main__":
    main()
