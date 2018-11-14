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
        X {list} -- 2d list object with int or float

    Returns:
        list -- 2d list object with float
    """

    m = len(X[0])
    x_max = [-float('inf') for _ in range(m)]
    x_min = [float('inf') for _ in range(m)]
    for row in X:
        x_max = [max(a, b) for a, b in zip(x_max, row)]
        x_min = [min(a, b) for a, b in zip(x_min, row)]
    ret = []
    for row in X:
        tmp = [(x - b) / (a - b) for a, b, x in zip(x_max, x_min, row)]
        ret.append(tmp)
    return ret
