# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-14 11:00:21
@Last Modified by:   tushushu
@Last Modified time: 2018-11-14 11:00:21
"""


def min_max_scale(X):
    """Scale the element of X into an interval [0, 1].

    Arguments:
        X {ndarray} -- 2d array object with int or float

    Returns:
        ndarray -- 2d array object with float
    """

    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    return (X - X_min) / (X_max - X_min)
