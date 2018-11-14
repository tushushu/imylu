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

from imylu.ensemble.gbdt_classifier import GradientBoostingClassifier
from imylu.utils.load_data import load_breast_cancer
from imylu.utils.model_selection import train_test_split, model_evaluation
from imylu.utils.utils import run_time


@run_time
def main():
    print("Tesing the performance of GBDT classifier...")
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)
    # Train model
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train, n_estimators=2,
            lr=0.8, max_depth=3, min_samples_split=2)
    # Model evaluation
    model_evaluation(clf, X_test, y_test)


if __name__ == "__main__":
    main()
