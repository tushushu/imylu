# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 14:33:11
@Last Modified by:   tushushu
@Last Modified time: 2019-05-22 15:41:11
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
    """Tesing the performance of GBDT classifier"""

    print("Tesing the performance of GBDT classifier...")
    # Load data
    data, label = load_breast_cancer()
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=20)
    # Train model
    clf = GradientBoostingClassifier()
    clf.fit(data_train, label_train, n_estimators=2,
            learning_rate=0.8, max_depth=3, min_samples_split=2)
    # Model evaluation
    model_evaluation(clf, data_test, label_test)


if __name__ == "__main__":
    main()
