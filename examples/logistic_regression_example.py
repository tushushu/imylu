# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:16:29
@Last Modified by:   tushushu
@Last Modified time: 2019-05-03 21:14:29
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.linear_model.logistic_regression import LogisticRegression
from imylu.utils.load_data import load_breast_cancer
from imylu.utils.model_selection import train_test_split, model_evaluation
from imylu.utils.preprocessing import min_max_scale
from imylu.utils.utils import run_time


def main():
    """Tesing the performance of LogisticRegression.
    """
    @run_time
    def batch():
        print("Tesing the performance of LogisticRegression(batch)...")
        # Train model
        clf = LogisticRegression()
        clf.fit(X=data_train, y=label_train, lr=0.1, epochs=1000)
        # Model evaluation
        model_evaluation(clf, data_test, label_test)
        print(clf)

    @run_time
    def stochastic():
        print("Tesing the performance of LogisticRegression(stochastic)...")
        # Train model
        clf = LogisticRegression()
        clf.fit(X=data_train, y=label_train, lr=0.01, epochs=100,
                method="stochastic", sample_rate=0.8)
        # Model evaluation
        model_evaluation(clf, data_test, label_test)
        print(clf)

    # Load data
    data, label = load_breast_cancer()
    data = min_max_scale(data)
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, random_state=10)
    batch()
    print()
    stochastic()


if __name__ == "__main__":
    main()
