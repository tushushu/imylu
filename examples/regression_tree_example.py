# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:34:18
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 17:34:18
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.utils import load_boston_house_prices, train_test_split, get_r2, run_time
from imylu.tree.regression_tree import RegressionTree


@run_time
def main():
    print("Tesing the accuracy of RegressionTree...")
    # Load data
    X, y = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10)
    # Train model
    reg = RegressionTree()
    reg.fit(X=X_train, y=y_train, max_depth=3)
    # Show rules
    reg.rules
    # Model accuracy
    get_r2(reg, X_test, y_test)


if __name__ == "__main__":
    main()
