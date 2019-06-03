# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:16:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 17:16:29
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.linear_model.ridge import Ridge
from imylu.utils.load_data import load_boston_house_prices
from imylu.utils.model_selection import get_r2, train_test_split
from imylu.utils.preprocessing import min_max_scale
from imylu.utils.utils import run_time


@run_time
def main():
    """Tesing the performance of Ridge Regressor(stochastic)
    """
    print("Tesing the performance of Ridge Regressor(stochastic)...")
    # Load data
    data, label = load_boston_house_prices()
    data = min_max_scale(data)
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=10)
    # Train model
    reg = Ridge()
    reg.fit(data=data_train, label=label_train, learning_rate=0.001, epochs=1000,
            alpha=1e-7, method="stochastic", sample_rate=0.5, random_state=10)
    # Model evaluation
    get_r2(reg, data_test, label_test)


if __name__ == "__main__":
    main()
