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

from imylu.utils import load_boston_house_prices, train_test_split, get_r2, \
    run_time, min_max_scale
from imylu.linear_model.ridge import Ridge


@run_time
def main():
    print("Tesing the accuracy of Ridge Regressor(stochastic)...")
    # Load data
    X, y = load_boston_house_prices()
    X = min_max_scale(X)
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    reg = Ridge()
    reg.fit(X=X_train, y=y_train, lr=0.001, epochs=1000,
            method="stochastic", sample_rate=0.5, alpha=1e-7)
    # Model accuracy
    get_r2(reg, X_test, y_test)


if __name__ == "__main__":
    main()
