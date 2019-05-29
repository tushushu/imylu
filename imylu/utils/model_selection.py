# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-14 11:02:02
@Last Modified by:   tushushu
@Last Modified time: 2018-11-14 11:02:02
"""
from itertools import chain
import numpy as np
from numpy.random import choice, seed


def train_test_split(data, label=None, prob=0.7, random_state=None):
    """Split data, label into train set and test set.

    Arguments:
        data {array} -- Training data.

    Keyword Arguments:
        label {array} -- Target values.
        prob {float} -- Train data expected rate between 0 and 1.
        (default: {0.7})
        random_state {int} -- Random seed. (default: {None})

    Returns:
        data_train {array}
        data_test {array}
        label_train {array}
        y_test {array}
    """

    # Set random state.
    if random_state is not None:
        seed(random_state)

    # Split data
    n_rows, _ = data.shape
    k = int(n_rows * prob)
    train_indexes = choice(range(n_rows), size=k, replace=False)
    test_indexes = np.array([i for i in range(n_rows) if i not in train_indexes])
    data_train = data[train_indexes]
    data_test = data[test_indexes]

    # Split label.
    if label is not None:
        label_train = label[train_indexes]
        label_test = label[test_indexes]
        ret = (data_train, data_test, label_train, label_test)
    else:
        ret = (data_train, data_test)

    # Cancel random state.
    if random_state is not None:
        seed(None)

    return ret


def get_r2(reg, X, y):
    """Calculate the goodness of fit of regression model.

    Arguments:
        reg {model} -- regression model.
        X {array} -- 2d array object with int or float.
        y {array} -- 1d array object with int.

    Returns:
        float
    """

    y_hat = reg.predict(X)
    r2 = _get_r2(y, y_hat)
    print("Test r2 is %.3f!\n" % r2)
    return r2


def model_evaluation(clf, X, y):
    """Calculate the prediction accuracy, recall, precision and auc
    of classification model.

    Arguments:
        clf {model} -- classification model.
        X {array} -- 2d array object with int or float.
        y {array} -- 1d array object with int.

    Returns:
        dict
    """

    y_hat = clf.predict(X)
    y_hat_prob = clf.predict_prob(X)

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
        y {array} -- 1d array object with int.
        y_hat {array} -- 1d array object with int.

    Returns:
        float
    """

    m = y.shape[0]
    n = y_hat.shape[0]
    assert m == n, "Lengths of two arrays do not match!"
    assert m != 0, "Empty array!"

    sse = ((y - y_hat) ** 2).mean()
    sst = y.var()
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
        y {array} -- 1d array object with int.
        y_hat {array} -- 1d array object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    return (y == y_hat).sum() / len(y)


def _get_precision(y, y_hat):
    """Calculate the prediction precision.

    Arguments:
        y {array} -- 1d array object with int.
        y_hat {array} -- 1d array object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    true_positive = (y * y_hat).sum()
    predicted_positive = y_hat.sum()
    return true_positive / predicted_positive


def _get_recall(y, y_hat):
    """Calculate the prediction recall.

    Arguments:
        y {array} -- 1d array object with int.
        y_hat {array} -- 1d array object with int.

    Returns:
        float
    """

    return _get_tpr(y, y_hat)


def _get_tpr(y, y_hat):
    """Calculate the prediction TPR.

    Arguments:
        y {array} -- 1d array object with int.
        y_hat {array} -- 1d array object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    true_positive = (y * y_hat).sum()
    actual_positive = y.sum()
    return true_positive / actual_positive


def _get_tnr(y, y_hat):
    """Calculate the prediction TNR.

    Arguments:
        y {array} -- 1d array object with int.
        y_hat {array} -- 1d array object with int.

    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    true_negative = ((1 - y) * (1 - y_hat)).sum()
    actual_negative = len(y) - y.sum()
    return true_negative / actual_negative


def _get_auc(y, y_hat_prob):
    """Calculate the prediction AUC.

    Arguments:
        y {array} -- 1d array object with int.
        y_hat_prob {array} -- 1d array object with int.

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
        y {array} -- 1d array object with int.
        y_hat_prob {array} -- 1d array object with int.

    Returns:
        array
    """

    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = (y_hat_prob >= threshold).astype(int)
        ret.append([_get_tpr(y, y_hat), 1 - _get_tnr(y, y_hat)])
    return ret
