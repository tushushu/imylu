# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:29:45
@Last Modified by:   tushushu
@Last Modified time: 2019-05-02 18:50:45
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
    """Tesing the performance of Gaussian NaiveBayes.
    """

    print("Tesing the performance of Gaussian NaiveBayes...")
    # Load data
    data, label = load_breast_cancer()
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, random_state=100)
    # Train model
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    # Model evaluation
    y_hat = clf.predict(data_test)
    acc = _get_acc(label_test, y_hat)
    print("Accuracy is %.3f" % acc)


if __name__ == "__main__":
    main()
