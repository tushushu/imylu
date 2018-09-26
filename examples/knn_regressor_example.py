# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 14:33:11
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 14:33:11
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.utils import get_r2, load_boston_house_prices, run_time, \
    train_test_split
from imylu.neighbors.knn_regressor import KNeighborsRegressor


@run_time
def main():
    print("Tesing the accuracy of KNN regressor...")
    # Load data
    X, y = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10)
    # Train model
    reg = KNeighborsRegressor()
    reg.fit(X=X_train, y=y_train, k_neighbors=3)
    # Model accuracy
    get_r2(reg, X_test, y_test)


if __name__ == "__main__":
    main()
