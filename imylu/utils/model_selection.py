# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-14 11:02:02
@Last Modified by:   tushushu
@Last Modified time: 2018-11-14 11:02:02
"""
from random import random, seed
from itertools import chain


def train_test_split(X, y, prob=0.7, random_state=None):
    """Split X, y into train set and test set.

    Arguments:
        X {list} -- 2d list object with int or float.
        y {list} -- 1d list object with int or float.

    Keyword Arguments:
        prob {float} -- Train data expected rate between 0 and 1.
        (default: {0.7})
        random_state {int} -- Random seed. (default: {None})

    Returns:
        X_train {list} -- 2d list object with int or float.
        X_test {list} -- 2d list object with int or float.
        y_train {list} -- 1d list object with int 0 or 1.
        y_test {list} -- 1d list object with int 0 or 1.
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


def get_r2(reg, X, y):
    """Calculate the goodness of fit of regression model.

    Arguments:
        reg {model} -- regression model.
        X {list} -- 2d list object with int or float.
        y {list} -- 1d list object with int.

    Returns:
        float
    """

    y_hat = reg.predict(X)
    r2 = _get_r2(y, y_hat)
    print("Test r2 is %.3f!" % r2)
    return r2


def model_evaluation(clf, X, y):
    """Calculate the prediction accuracy, recall, precision and auc
    of classification model.

    Arguments:
        clf {model} -- classification model.
        X {list} -- 2d list object with int or float.
        y {list} -- 1d list object with int.

    Returns:
        dict
    """

    y_hat = clf.predict(X)
    y_hat_prob = [clf._predict(Xi) for Xi in X]

    ret = dict()
    ret["Accuracy"] = _get_acc(y, y_hat)
    ret["Recall"] = _get_recall(y, y_hat)
    ret["Precision"] = _get_precision(y, y_hat)
    ret["AUC"] = _get_auc(y, y_hat_prob)

    for k, v in ret.items():
        print("%s: %.3f" % (k, v))
    print()
    return ret


def _get_r2(y, y_hat):
    """Calculate the goodness of fit.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat {list} -- 1d list object with int.

    Returns:
        float
    """

    m = len(y)
    n = len(y_hat)
    assert m == n, "Lengths of two arrays do not match!"
    assert m != 0, "Empty array!"

    sse = sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y, y_hat))
    y_avg = sum(y) / len(y)
    sst = sum((yi - y_avg) ** 2 for yi in y)
    r2 = 1 - sse / sst
    return r2


def _clf_input_check(y, y_hat):
    m = len(y)
    n = len(y_hat)
    elements = chain(y, y_hat)
    valid_elements = {0, 1}
    assert m == n, "Lengths of two arrays do not match!"
    assert m != 0, "Empty array!"
    assert all(element in valid_elements
               for element in elements), "Array values have to be 0 or 1!"


def _get_acc(y, y_hat):
    """Calculate the prediction accuracy.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat {list} -- 1d list object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)


def _get_precision(y, y_hat):
    """Calculate the prediction precision.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat {list} -- 1d list object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    predicted_positive = sum(y_hat)
    return true_positive / predicted_positive


def _get_recall(y, y_hat):
    """Calculate the prediction recall.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat {list} -- 1d list object with int.

    Returns:
        float
    """

    return _get_tpr(y, y_hat)


def _get_tpr(y, y_hat):
    """Calculate the prediction TPR.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat {list} -- 1d list object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_positive / actual_positive


def _get_tnr(y, y_hat):
    """Calculate the prediction TNR.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat {list} -- 1d list object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
    actual_negative = len(y) - sum(y)
    return true_negative / actual_negative


def _get_auc(y, y_hat_prob):
    """Calculate the prediction AUC.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat_prob {list} -- 1d list object with int.

    Returns:
        float
    """

    roc = iter(_get_roc(y, y_hat_prob))
    tpr_pre, fpr_pre = next(roc)
    auc = 0
    for tpr, fpr in roc:
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
        tpr_pre = tpr
        fpr_pre = fpr
    return auc


def _get_roc(y, y_hat_prob):
    """Calculate the points of ROC.

    Arguments:
        y {list} -- 1d list object with int.
        y_hat_prob {list} -- 1d list object with int.

    Returns:
        list
    """

    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
        ret.append([_get_tpr(y, y_hat), 1 - _get_tnr(y, y_hat)])
    return ret
