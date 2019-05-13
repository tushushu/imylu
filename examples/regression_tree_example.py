# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:34:18
@Last Modified by:   tushushu
@Last Modified time: 2019-05-04 17:34:18
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.tree.regression_tree import RegressionTree
from imylu.utils.load_data import load_boston_house_prices
from imylu.utils.model_selection import get_r2, train_test_split
from imylu.utils.utils import run_time


@run_time
def main():
    """Tesing the performance of RegressionTree
    """
    print("Tesing the performance of RegressionTree...")
    # Load data
    data, label = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, random_state=200)
    # Train model
    reg = RegressionTree()
    reg.fit(data=data_train, label=label_train, max_depth=5)
    # Show rules
    print(reg)
    # Model evaluation
    get_r2(reg, data_test, label_test)


if __name__ == "__main__":
    main()
