# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 14:33:11
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 14:33:11
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

import imylu.utils
from imylu.utils import get_acc, load_breast_cancer, run_time, train_test_split
from imylu.ensemble.gbdt_classifier import GradientBoostingClassifier


@run_time
def main():

    print("Tesing the accuracy of GBDT classifier...")
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
