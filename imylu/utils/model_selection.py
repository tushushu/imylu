# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-14 11:02:02
@Last Modified by:   tushushu
@Last Modified time: 2018-11-14 11:02:02
"""
from random import random, seed


def train_test_split(X, y, prob=0.7, random_state=None):
    """Split X, y into train set and test set.

    Arguments:
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int or float

    Keyword Arguments:
        prob {float} -- Train data expected rate between 0 and 1
        (default: {0.7})
        random_state {int} -- Random seed (default: {None})

    Returns:
        X_train {list} -- 2d list object with int or float
        X_test {list} -- 2d list object with int or float
        y_train {list} -- 1d list object with int 0 or 1
        y_test {list} -- 1d list object with int 0 or 1
    """

    if random_state is not None:
        seed(random_state)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(X)):
        if random() < prob:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    # Make the fixed random_state random again
    seed()
    return X_train, X_test, y_train, y_test


def get_acc(clf, X, y):
    """Calculate the prediction accuracy of classification model.

    Arguments:
        clf {model} -- classification model
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int

    Returns:
        float
    """

    acc = sum((yi_hat == yi for yi_hat, yi in zip(clf.predict(X), y))) / len(y)
    print("Test accuracy is %.3f%%!" % (acc * 100))
    return acc


def get_r2(reg, X, y):
    """Calculate the goodness of fit of regression model.

    Arguments:
        reg {model} -- regression model
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int

    Returns:
        float
    """

    sse = sum((yi_hat - yi) ** 2 for yi_hat, yi in zip(reg.predict(X), y))
    y_avg = sum(y) / len(y)
    sst = sum((yi - y_avg) ** 2 for yi in y)
    r2 = 1 - sse / sst
    print("Test r2 is %.3f!" % r2)
    return r2
