# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-10-15 15:19:14
@Last Modified by:   tushushu
@Last Modified time: 2018-10-15 15:19:14
"""
from collections import defaultdict
from .matrix import Matrix
from random import random


class ALS(object):
    """Alternative least square class.

    Attributes:
        user_id_to_col_no {dict} -- Look up user id by matrix column number.
        Item_id_to_col_no {dict} -- Look up item id by matrix column number.
        ratings {dict} -- 2D dict to keep the item rating data.
        mse {float} -- Mean square error.
    """

    def __init__(self):
        self.user_id_to_col_no = None
        self.item_id_to_col_no = None
        self.user_matrix = None
        self.item_matrix = None
        self.mse = None

    def _process_data(self, X):
        """Transform the item rating data into a sparse matrix.

        Arguments:
            X {list} -- 2d list with int or float(user_id, item_id, rating)

        Returns:
            tuple -- Distinct user_ids.
            tuple -- Distinct item_ids.
            dict -- The items ratings by users. {user_id: {item_id: rating}}
            tuple -- Shape of sparse matrix.
        """

        # Look up matrix column number by user id.
        user_id_to_col_no = tuple((set(map(lambda x: x[0], X))))
        # Look up user id by matrix column number.
        col_no_to_user_id = dict(enumerate(user_id_to_col_no))
        # Look up matrix column number by item id.
        item_id_to_col_no = tuple((set(map(lambda x: x[1], X))))
        # Look up item id by matrix column number.
        col_no_to_item_id = dict(enumerate(item_id_to_col_no))
        # The shape of item rating data matrix.
        shape = (len(user_id_to_col_no), len(item_id_to_col_no))
        # Sparse matrix and its inverse of item rating data.
        ratings = defaultdict(lambda: defaultdict(int))
        ratings_T = defaultdict(lambda: defaultdict(int))
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating
        ret = (user_id_to_col_no, col_no_to_user_id, item_id_to_col_no,
               col_no_to_item_id, shape, ratings, ratings_T)
        return ret

    def _users_mul_ratings(self, mat, ratings):
        assert mat.is_square, "The matrix has to be square!"
        raise NotImplementedError

    def _gen_random_matrix(shape):
        """Generate a matrix with random values.

        Arguments:
            shape {tuple} -- The shape of matrix.

        Returns:
            Matrix
        """

        m, n = shape
        data = [[random() for _ in range(n)] for _ in range(m)]
        return Matrix(data)

    def _get_mse(self, ratings, users, items):
        """MSE = Sum((R - U_T * I)) ^ 2 / n_elements

        Arguments:
            ratings {dict} -- The items ratings by users.
            users {Matrix} -- k * m matrix.
            items {Matrix} -- k * n matrix.
        """

        ratings_hat = users.mat_mul(items)
        m, n = ratings_hat.shape
        mse = 0.0
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                item_id = self.item_ids[j]
                if ratings[user_id][item_id] > 0:
                    rating = ratings[user_id][item_id]
                    rating_hat = ratings_hat[i][j]
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error
        n_elements = sum(map(len, ratings))
        return mse / n_elements

    def fit(self, X, k, max_iter=100):
        """Build an ALS model.
        Suppose the rating matrix R can be decomposed as U * I,
        U stands for User and I stands for Item.
        R(m, n) = U(k, m)_transpose * I(k, n)

        Use MSE as loss function,
        Loss(U, I) = sum((R_ij - U_i_transpose * I_j) ^ 2)

        Take the partial of the function,
        dLoss(U, I) / dU = -2 * sum(I_j *
        (R_ij - U_i_transpose * I_j)_transpose)

        Let dLoss(U, I) / dU = 0, then
        I * R_transpose - I * I_transpose * U = 0
        U = (I * I_transpose) ^ (-1) * I * R_transpose

        Same logic,
        I = (U * U_transpose) ^ (-1) * U * R


        Arguments:
            X {list} -- 2d list with int or float(user_id, item_id, rating)
            k {int} -- The rank of user and item matrix.

        Keyword Arguments:
            max_iter {int} -- Maximum numbers of iteration. (default: {100})
        """

        # Process item rating data.
        self.user_id_to_col_no, col_no_to_user_id,\
            self.item_id_to_col_no, col_no_to_item_id,\
            shape, ratings, ratings_T = self._process_data(X)
        m, n = shape
        error_msg = "Parameter k must be less than the rank of original matrix"
        assert k < min(m, n), error_msg
        # Initialize users and items matrix.
        users = self._gen_random_matrix((k, m))
        items = None
        # Minimize the MSE by EM algorithms.
        for i in range(max_iter):
            if i % 2:
                # U = (I * I_transpose) ^ (-1) * I * R_transpose
                users = self._mul_ratings(
                    items.mat_mul(items.transpose).inverse.mat_mul(items),
                    ratings_T,
                    col_no_to_user_id
                )
            else:
                # I = (U * U_transpose) ^ (-1) * U * R
                items = self._mul_ratings(
                    users.mat_mul(users.transpose).inverse.mat_mul(users),
                    ratings,
                    col_no_to_item_id
                )
            mse = self._get_mse(ratings, users, items)
            print("Iterations: %d, MSE: %.3f" % (i + 1, mse))
        # Final MSE.
        self.mse = mse

    def _predict(self, user_id):
        """Predict the items ratings by user.

        Arguments:
            user_id {int}

        Returns:
            list -- [(item_id, score), ..., (item_id, score)]
        """
        raise NotImplementedError

    def predict(self, user_ids):
        """Predict the items ratings by users.

        Arguments:
            user_ids {list} -- 1d list with int.

        Returns:
            list -- 2d list with item_id and score.
        """

        return [self._predict(user_id) for user_id in user_ids]
