import os
from random import random, seed
from time import time
from math import exp


def load_data(file_name):
    """Read csv file

    Arguments:
        file_name {str} -- csv file name

    Returns:
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int or float
    """

    path = os.path.join(os.getcwd(), "dataset", "%s.csv" % file_name)
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
    return load_data("breast_cancer")


def load_boston_house_prices():
    return load_data("boston_house_prices")


def min_max_scale(X):
    """Scale the element of X into an interval [0, 1]

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
        tmp = [(x - b)/(a - b) for a, b, x in zip(x_max, x_min, row)]
        ret.append(tmp)
    return ret


def train_test_split(X, y, prob=0.7, random_state=None):
    """Split X, y into train set and test set.

    Arguments:
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int or float

    Keyword Arguments:
        prob {float} -- Train data expected rate between 0 and 1 (default: {0.7})
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
    acc = sum((yi_hat == yi for yi_hat, yi in zip(clf.predict(X), y))) / len(y)
    print("Test accuracy is %.3f%%!" % (acc * 100))
    return acc


def run_time(func):
    def wrapper():
        start = time()
        func()
        print("Total run time is %.2f s" % (time() - start))
    return wrapper


def get_r2(reg, X, y):
    sse = sum((yi_hat - yi) ** 2 for yi_hat, yi in zip(reg.predict(X), y))
    y_avg = sum(y) / len(y)
    sst = sum((yi - y_avg) ** 2 for yi in y)
    r2 = 1 - sse/sst
    print("Test r2 is %.3f!" % r2)
    return r2


def sigmoid(x, x_min=-100):
    """Sigmoid(x) = 1 / (1 + e^(-x))

    Arguments:
        x {float}

    Keyword Arguments:
        x_min {int} -- It would cause math range error when x < -709 (default: {-100})

    Returns:
        float -- between 0 and 1
    """

    return 1 / (1 + exp(-x)) if x > x_min else 0
