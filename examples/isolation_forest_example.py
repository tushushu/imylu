# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 16:25:09
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 16:25:09
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.ensemble.isolation_forest import IsolationForest
from imylu.utils.utils import run_time
from random import random


@run_time
def main():
    print("Comparing average score of X and outlier's score...")
    # Generate a dataset randomly
    n = 100
    X = [[random() for _ in range(5)] for _ in range(n)]
    # Add outliers
    X.append([10] * 5)
    # Train model
    clf = IsolationForest()
    clf.fit(X, n_samples=500)
    # Show result
    print("Average score is %.2f" % (sum(clf.predict(X)) / len(X)))
    print("Outlier's score is %.2f" % clf._predict(X[-1]))


if __name__ == "__main__":
    main()
