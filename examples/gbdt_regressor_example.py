# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 14:33:11
@Last Modified by:   tushushu
@Last Modified time: 2019-05-22 11:10:11
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.ensemble.gbdt_regressor import GradientBoostingRegressor
from imylu.utils.load_data import load_boston_house_prices
from imylu.utils.model_selection import get_r2, train_test_split
from imylu.utils.utils import run_time


@run_time
def main():
    print("Tesing the performance of GBDT regressor...")
    # Load data
    data, label = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, random_state=10)
    # Train model
    reg = GradientBoostingRegressor()
    reg.fit(data=data_train, label=label_train, n_estimators=4,
            learning_rate=0.5, max_depth=3, min_samples_split=2)
    # Model evaluation
    get_r2(reg, data_test, label_test)


if __name__ == "__main__":
    main()
