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

from imylu.linear_model.linear_regression import LinearRegression
from imylu.utils.load_data import load_boston_house_prices
from imylu.utils.model_selection import get_r2, train_test_split
from imylu.utils.preprocessing import min_max_scale
from imylu.utils.utils import run_time


def main():
    @run_time
    def batch():
        print("Tesing the performance of LinearRegression(batch)...")
        # Train model
        reg = LinearRegression()
        reg.fit(X=X_train, y=y_train, lr=0.02, epochs=5000)
        # Model evaluation
        get_r2(reg, X_test, y_test)

    @run_time
    def stochastic():
        print("Tesing the performance of LinearRegression(stochastic)...")
        # Train model
        reg = LinearRegression()
        reg.fit(X=X_train, y=y_train, lr=0.001, epochs=1000,
                method="stochastic", sample_rate=0.5)
        # Model evaluation
        get_r2(reg, X_test, y_test)

    # Load data
    X, y = load_boston_house_prices()
    X = min_max_scale(X)
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    batch()
    stochastic()


if __name__ == "__main__":
    main()
