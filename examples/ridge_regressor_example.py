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
    print("Tesing the performance of Ridge Regressor(stochastic)...")
    # Load data
    X, y = load_boston_house_prices()
    X = min_max_scale(X)
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    reg = Ridge()
    reg.fit(X=X_train, y=y_train, lr=0.001, epochs=1000,
            method="stochastic", sample_rate=0.5, alpha=1e-7)
    # Model evaluation
    get_r2(reg, X_test, y_test)


if __name__ == "__main__":
    main()
