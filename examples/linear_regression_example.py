# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2019-01-07 14:28:00
"""

import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))


from imylu.utils.utils import run_time
from imylu.utils.preprocessing import min_max_scale
from imylu.utils.model_selection import get_r2, train_test_split
from imylu.utils.load_data import load_boston_house_prices
from imylu.linear_model.linear_regression import LinearRegression


def main():
    """Tesing the performance of LinearRegression.
    """
    @run_time
    def batch():
        print("Tesing the performance of LinearRegression(batch)...")
        # Train model
        reg = LinearRegression()
        reg.fit(data=data_train, label=label_train, learning_rate=0.1, epochs=1000)
        # Model evaluation
        get_r2(reg, data_test, label_test)
        print(reg)

    @run_time
    def stochastic():
        print("Tesing the performance of LinearRegression(stochastic)...")
        # Train model
        reg = LinearRegression()
        reg.fit(data=data_train, label=label_train, learning_rate=0.05, epochs=50,
                method="stochastic", sample_rate=0.6)
        # Model evaluation
        get_r2(reg, data_test, label_test)
        print(reg)

    # Load data
    data, label = load_boston_house_prices()
    data = min_max_scale(data)
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, random_state=20)
    batch()
    stochastic()


if __name__ == "__main__":
    main()
