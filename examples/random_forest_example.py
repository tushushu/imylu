# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 16:39:58
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 16:39:58
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.ensemble.random_forest import RandomForest
from imylu.utils.load_data import load_breast_cancer
from imylu.utils.model_selection import train_test_split, model_evaluation
from imylu.utils.utils import run_time


@run_time
def main():
    """Tesing the performance of RandomForest...
    """
    print("Tesing the performance of RandomForest...")
    # Load data
    data, label = load_breast_cancer()
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=40)

    # Train model
    clf = RandomForest()
    clf.fit(data_train, label_train, n_estimators=50, max_depth=5, random_state=10)
    # Model evaluation
    model_evaluation(clf, data_test, label_test)


if __name__ == "__main__":
    main()
