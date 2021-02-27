# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-14 11:11:35
@Last Modified by:   tushushu
@Last Modified time: 2018-11-14 11:11:35
"""

from copy import copy
from itertools import tee
from numpy import exp, ndarray
from random import randint
from statistics import median
from time import time
from typing import List


def arr2str(arr: ndarray, n_digits: int) -> str:
    ret = ", ".join(map(lambda x: str(round(x, n_digits)), arr))
    return "[%s]" % ret


def run_time(fn):
    """Decorator for calculating function runtime.Depending on the length of time,
    seconds, milliseconds, microseconds or nanoseconds are used.

    Arguments:
        fn {function}

    Returns:
        function
    """

    def inner():
        start = time()
        fn()
        ret = time() - start
        if ret < 1e-6:
            unit = "ns"
            ret *= 1e9
        elif ret < 1e-3:
            unit = "us"
            ret *= 1e6
        elif ret < 1:
            unit = "ms"
            ret *= 1e3
        else:
            unit = "s"
        print("Total run time is %.1f %s\n" % (ret, unit))
    return inner


def sigmoid(x):
    """Calculate the sigmoid value of x.
    Sigmoid(x) = 1 / (1 + e^(-x))
    It would cause math range error when x < -709

    Arguments:
        x {float}

    Returns:
        float -- between 0 and 1
    """

    return 1 / (1 + exp(-x))


def split_list(X, idxs, feature, split, low, high):
    """ Sort the list, if the element in the array is less than result index,
    the element value is less than the split. Otherwise, the element value is
    equal to or greater than the split.

    Arguments:
        X {list} -- 2d list object with int or float
        idx {list} -- indexes, 1d list object with int
        feature {int} -- Feature number
        split {float} -- The split point value

    Returns:
        int -- index
    """

    p = low
    q = high - 1
    while p <= q:
        if X[idxs[p]][feature] < split:
            p += 1
        elif X[idxs[q]][feature] >= split:
            q -= 1
        else:
            idxs[p], idxs[q] = idxs[q], idxs[p]
    return p


def list_split(X, idxs, feature, split):
    """Another implementation of "split_list" function for performance comparison.

    Arguments:
        nums {list} -- 1d list with int or float
        split {float} -- The split point value

    Returns:
        list -- 2d list with left and right split result
    """

    ret = [[], []]
    while idxs:
        if X[idxs[0]][feature] < split:
            ret[0].append(idxs.pop(0))
        else:
            ret[1].append(idxs.pop(0))
    return ret


def _test_split_list(iterations=10**4, max_n_samples=1000, max_n_features=10,
                     max_element_value=100):
    """Test correctness and runtime efficiency of both split_list functions.
    _split_list takes about 2.4 times as split_list does.

    Keyword Arguments:
        iterations {int} -- How many times to iterate. (default: {10**4})
        max_arr_len {int} -- Max random length of array (default: {1000})
        max_num {int} -- Max value of array's elements (default: {100})
    """

    time_1 = time_2 = 0
    for _ in range(iterations):
        n = randint(1, max_n_samples)
        m = randint(1, max_n_features)
        X = [[randint(1, max_element_value) for _ in range(m)]
             for _ in range(n)]
        idxs_1 = list(range(n))
        idxs_2 = copy(idxs_1)
        feature = randint(1, m) - 1
        split = median(map(lambda i: X[i][feature], range(n)))
        low = 0
        high = n

        start = time()
        ret_1 = split_list(X, idxs_1, feature, split, low, high)
        time_1 += time() - start

        start = time()
        ret_2 = list_split(X, idxs_2, feature, split)
        time_2 += time() - start

        assert all(i_1 == i_2 for i_1, i_2 in zip(
            sorted(idxs_1[low:ret_1]), sorted(ret_2[0])))
        assert all(i_1 == i_2 for i_1, i_2 in zip(
            sorted(idxs_1[ret_1:high]), sorted(ret_2[1])))

    print("Test passed!")
    print("split_list runtime for %d iterations  is: %.3f seconds" %
          (iterations, time_1))
    print("_split_list runtime for %d iterations  is: %.3f seconds" %
          (iterations, time_2))


def get_euclidean_distance(arr1: ndarray, arr2: ndarray) -> float:
    """"Calculate the Euclidean distance of two vectors.

    Arguments:
        arr1 {ndarray}
        arr2 {ndarray}

    Returns:
        float
    """

    return ((arr1 - arr2) ** 2).sum() ** 0.5

def get_eu_dist(arr1: List, arr2: List) -> float:
    """Calculate the Euclidean distance of two vectors.
    Arguments:
        arr1 {list} -- 1d list object with int or float
        arr2 {list} -- 1d list object with int or float
    Returns:
        float -- Euclidean distance
    """

    return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5


def get_cosine_distance(arr1, arr2):
    """Calculate the cosine distance of two vectors.
    Arguments:
        arr1 {list} -- 1d list object with int or float
        arr2 {list} -- 1d list object with int or float
    Returns:
        float -- cosine distance
    """
    numerator = sum(x1 * x2 for x1, x2 in zip(arr1, arr2))
    denominator = (sum(x1 ** 2 for x1 in arr1) *
                   sum(x2 ** 2 for x2 in arr2)) ** 0.5
    return numerator / denominator


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ...

    Arguments:
        iterable {iterable}

    Returns:
        zip
    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def arg_max_2d(dic):
    return max(((k, *max(dic_inner.items(), key=lambda x: x[1]))
                for k, dic_inner in dic.items()), key=lambda x: x[2])[:2]
