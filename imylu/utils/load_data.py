# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2019-01-07 14:28:52
"""

import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
BASE_PATH = os.path.abspath("..")

import numpy as np


def _load_data(file_name):
    """Read csv file.

    Arguments:
        file_name {str} -- csv file name

    Returns:
        X {array} -- 2d array object with int or float
        y {array} -- 1d array object with int or float
    """

    path = os.path.join(BASE_PATH, "dataset", "%s.csv" % file_name)
    data = np.loadtxt(path, delimiter=',')
    X, y = data[:, ::-1], data[:, -1]
    return X, y


def load_breast_cancer():
    """Load breast cancer data for classification.

    Returns:
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int or float
    """

    return _load_data("breast_cancer")


def load_boston_house_prices():
    """Load boston house prices data for regression.

    Returns:
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int or float
    """

    return _load_data("boston_house_prices")


def load_tagged_speech():
    """Load tagged speech data for classification.

    Returns:
        X {list} -- 2d list object with str.
        y {list} -- 1d list object with str.
    """

    file_names = ["observations", "states"]

    def data_process(file_name):
        path = os.path.join(BASE_PATH, "dataset", "%s.csv" % file_name)
        f = open(path)
        data = [line[:-1].split("|") for line in f]
        f.close()
        return data
    return [data_process(file_name) for file_name in file_names]


def load_movie_ratings():
    """Load movie ratings data for recommedation.

    Returns:
        list -- userId, movieId, rating
    """

    file_name = "movie_ratings"
    path = os.path.join(BASE_PATH, "dataset", "%s.csv" % file_name)
    f = open(path)
    lines = iter(f)
    col_names = ", ".join(next(lines)[:-1].split(",")[:-1])
    print("The column names are: %s." % col_names)
    data = [[float(x) if i == 2 else int(x)
             for i, x in enumerate(line[:-1].split(",")[:-1])]
            for line in lines]
    f.close()
    return data
