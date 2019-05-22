# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-05 17:37:34
@Last Modified by: tushushu
@Last Modified time: 2018-07-05 17:37:34
"""
import numpy as np
from numpy import array

from .gbdt_base import GradientBoostingBase


class GradientBoostingRegressor(GradientBoostingBase):
    """Gradient Boosting Regressor"""

    def __init__(self):
        GradientBoostingBase.__init__(self)
        self.fn = lambda x: x

    def _get_init_val(self, label: array):
        """Calculate the initial prediction of y
        Set MSE as loss function, yi <- y, and c is a constant:
        L = MSE(y, c) = Sum((yi-c) ^ 2) / n

        Get derivative of c:
        dL / dc = Sum(-2 * (yi-c)) / n
        dL / dc = -2 * (Sum(yi) / n - Sum(c) / n)
        dL / dc = -2 * (Mean(yi) - c)

        Let derivative equals to zero, then we get initial constant value
        to minimize MSE:
        -2 * (Mean(yi) - c) = 0
        c = Mean(yi)
        ----------------------------------------------------------------------------------------

        Arguments:
            label {array} -- Target values.

        Returns:
            float
        """

        return label.mean()

    def _update_score(self, tree, data:array, y_hat, residuals):
        """update the score of regression tree leaf node
        Fm(xi) = Fm-1(xi) + fm(xi)

        Loss Function:
        Loss(yi, Fm(xi)) = Sum((yi - Fm(xi)) ^ 2) / n

        Taylor 1st:
        f(x + x_delta) = f(x) + f'(x) * x_delta
        f(x) = g'(x)
        g'(x + x_delta) = g'(x) + g"(x) * x_delta

        1st derivative:
        Loss'(yi, Fm(xi)) = -2 * Sum(yi - Fm(xi)) / n

        2nd derivative:
        Loss"(yi, Fm(xi)) = -2

        So,
        Loss'(yi, Fm(xi)) = Loss'(yi, Fm-1(xi) + fm(xi))
        = Loss'(yi, Fm-1(xi)) + Loss"(yi, Fm-1(xi)) *  fm(xi) = 0
        fm(xi) = - Loss'(yi, Fm-1(xi)) / Loss"(yi, Fm-1(xi))
        fm(xi) = -2 * Sum(yi - Fm-1(xi) / n / -2
        fm(xi) = Sum(yi - Fm-1(xi)) / n
        fm(xi) = Mean(yi - Fm-1(xi))
        ----------------------------------------------------------------------------------------

        Arguments:
            tree {RegressionTree}
            data {array} -- Training data.
            prediction {array} -- Prediction of label.
            residuals {array}
        """

        pass

    def predict(self, data:array)->array:
        """Get the prediction of label.

        Arguments:
            data {array} -- Training data.

        Returns:
            array -- Prediction of label.
        """

        return np.apply_along_axis(self._predict, axis=1, arr=data)
