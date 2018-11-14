# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:29:45
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 17:29:45
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.probability_model.gaussian_nb import GaussianNB
from imylu.utils.load_data import load_breast_cancer
from imylu.utils.model_selection import train_test_split, _get_acc
from imylu.utils.utils import run_time


@run_time
def main():
    print("Tesing the performance of Gaussian NaiveBayes...")
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    # Model evaluation
    y_hat = clf.predict(X_test)
    acc = _get_acc(y_test, y_hat)
    print("Accuracy is %.3f" % acc)


if __name__ == "__main__":
    main()
