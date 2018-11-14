# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-14 10:58:24
@Last Modified by:   tushushu
@Last Modified time: 2018-11-14 10:58:24
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
BASE_PATH = os.path.abspath("..")

from random import randint


def _load_data(file_name):
    """Read csv file.

    Arguments:
        file_name {str} -- csv file name

    Returns:
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int or float
    """

    path = os.path.join(BASE_PATH, "dataset", "%s.csv" % file_name)
    f = open(path)
    X = []
    y = []
    for line in f:
        line = line[:-1].split(",")
        xi = [float(s) for s in line[:-1]]
        yi = line[-1]
        if '.' in yi:
            yi = float(yi)
        else:
            yi = int(yi)
        X.append(xi)
        y.append(yi)
    f.close()
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


def gen_data(low, high, n_rows, n_cols=None):
    """Generate dataset randomly.

    Arguments:
        low {int} -- The minimum value of element generated.
        high {int} -- The maximum value of element generated.
        n_rows {int} -- Number of rows.
        n_cols {int} -- Number of columns.

    Returns:
        list -- 1d or 2d list with int
    """
    if n_cols is None:
        ret = [randint(low, high) for _ in range(n_rows)]
    else:
        ret = [[randint(low, high) for _ in range(n_cols)]
               for _ in range(n_rows)]
    return ret
