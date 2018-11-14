# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-10-15 15:19:14
@Last Modified by:   tushushu
@Last Modified time: 2018-10-15 15:19:14
"""
from collections import defaultdict
from ..utils.matrix import Matrix
from random import random


class ALS(object):
    """Alternative least square class.

    Attributes:
        user_ids {tuple} -- Look up user id by matrix column number.
        item_ids {tuple} -- Look up item id by matrix column number.
        user_ids_dict {dict} -- Look up matrix column number by user id.
        item_ids_dict {dict} -- Look up matrix column number by item id.
        user_matrix {Matrix} -- k * m matrix, m equals number of user_ids.
        item_matrix {Matrix} -- k * n matrix, n equals number of item_ids.
        user_items {dict} -- Store what items has been viewed by users.
        shape {tuple} -- Dimension of ratings matrix.
        rmse {float} -- Square root of mse,
        (Sum((R - U_T * I)) ^ 2 / n_elements) ^ 0.5.
    """

    def __init__(self):
        self.user_ids = None
        self.item_ids = None
        self.user_ids_dict = None
        self.item_ids_dict = None
        self.user_matrix = None
        self.item_matrix = None
        self.user_items = None
        self.shape = None
        self.rmse = None

    def _process_data(self, X):
        """Transform the item rating data into a sparse matrix.

        Arguments:
            X {list} -- 2d list with int or float(user_id, item_id, rating)

        Returns:
            dict -- The items ratings by users. {user_id: {item_id: rating}}
            dict -- The items ratings by users. {item_id: {user_id: rating}}
        """

        # Process user ids.
        self.user_ids = tuple((set(map(lambda x: x[0], X))))
        self.user_ids_dict = dict(map(lambda x: x[::-1],
                                      enumerate(self.user_ids)))

        # Process item ids.
        self.item_ids = tuple((set(map(lambda x: x[1], X))))
        self.item_ids_dict = dict(map(lambda x: x[::-1],
                                      enumerate(self.item_ids)))

        # The shape of item rating data matrix.
        self.shape = (len(self.user_ids), len(self.item_ids))

        # Sparse matrix and its inverse of item rating data.
        ratings = defaultdict(lambda: defaultdict(int))
        ratings_T = defaultdict(lambda: defaultdict(int))
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating

        # Result validation.
        err_msg = "Length of user_ids %d and ratings %d not match!" % (
            len(self.user_ids), len(ratings))
        assert len(self.user_ids) == len(ratings), err_msg

        err_msg = "Length of item_ids %d and ratings_T %d not match!" % (
            len(self.item_ids), len(ratings_T))
        assert len(self.item_ids) == len(ratings_T), err_msg
        return ratings, ratings_T

    def _users_mul_ratings(self, users, ratings_T):
        """Multiply a dense matrix(user matrix) with sparse matrix (rating matrix).
        The result(items) is a k * n matrix, n stands for number of item_ids.

        Arguments:
            users {Matrix} -- k * m matrix, m stands for number of user_ids.
            ratings_T {dict} -- The items ratings by users.
            {item_id: {user_id: rating}}

        Returns:
            Matrix -- Item matrix.
        """

        def f(users_row, item_id):
            user_ids = iter(ratings_T[item_id].keys())
            scores = iter(ratings_T[item_id].values())
            col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
            _users_row = map(lambda x: users_row[x], col_nos)
            return sum(a * b for a, b in zip(_users_row, scores))

        ret = [[f(users_row, item_id) for item_id in self.item_ids]
               for users_row in users.data]
        return Matrix(ret)

    def _items_mul_ratings(self, items, ratings):
        """Multiply a dense matrix(item matrix) with sparse matrix (rating matrix).
        The result(users) is a k * m matrix, m stands for number of user_ids.

        Arguments:
            items {Matrix} -- k * n matrix, n stands for number of item_ids.
            ratings {dict} -- The items ratings by users.
            {user_id: {item_id: rating}}

        Returns:
            Matrix -- User matrix.
        """

        def f(items_row, user_id):
            item_ids = iter(ratings[user_id].keys())
            scores = iter(ratings[user_id].values())
            col_nos = map(lambda x: self.item_ids_dict[x], item_ids)
            _items_row = map(lambda x: items_row[x], col_nos)
            return sum(a * b for a, b in zip(_items_row, scores))

        ret = [[f(items_row, user_id) for user_id in self.user_ids]
               for items_row in items.data]
        return Matrix(ret)

    def _gen_random_matrix(self, n_rows, n_colums):
        """Generate a n_rows * n_columns matrix with random numbers.

        Arguments:
            n_rows {int} -- The number of rows.
            n_colums {int} -- The number of columns.

        Returns:
            Matrix
        """

        data = [[random() for _ in range(n_colums)] for _ in range(n_rows)]
        return Matrix(data)

    def _get_rmse(self, ratings):
        """Calculate RMSE.

        Arguments:
            ratings {dict} -- The items ratings by users.

        Returns:
            float
        """

        m, n = self.shape
        mse = 0.0
        n_elements = sum(map(len, ratings.values()))
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                item_id = self.item_ids[j]
                rating = ratings[user_id][item_id]
                if rating > 0:
                    user_row = self.user_matrix.col(i).transpose
                    item_col = self.item_matrix.col(j)
                    rating_hat = user_row.mat_mul(item_col).data[0][0]
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error / n_elements
        return mse ** 0.5

    def fit(self, X, k, max_iter=10):
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
            max_iter {int} -- Maximum numbers of iteration. (default: {10})
        """

        # Process item rating data.
        ratings, ratings_T = self._process_data(X)
        # Store what items has been viewed by users.
        self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
        # Parameter validation.
        m, n = self.shape
        error_msg = "Parameter k must be less than the rank of original matrix"
        assert k < min(m, n), error_msg
        # Initialize users and items matrix.
        self.user_matrix = self._gen_random_matrix(k, m)
        # Minimize the RMSE by EM algorithms.
        for i in range(max_iter):
            if i % 2:
                # U = (I * I_transpose) ^ (-1) * I * R_transpose
                items = self.item_matrix
                self.user_matrix = self._items_mul_ratings(
                    items.mat_mul(items.transpose).inverse.mat_mul(items),
                    ratings
                )
            else:
                # I = (U * U_transpose) ^ (-1) * U * R
                users = self.user_matrix
                self.item_matrix = self._users_mul_ratings(
                    users.mat_mul(users.transpose).inverse.mat_mul(users),
                    ratings_T
                )
            rmse = self._get_rmse(ratings)
            print("Iterations: %d, RMSE: %.6f" % (i + 1, rmse))
        # Final RMSE.
        self.rmse = rmse

    def _predict(self, user_id, n_items):
        """Predict the items ratings by user.

        Arguments:
            user_id {int}

        Returns:
            list -- [(item_id, score), ..., (item_id, score)]
        """

        # Get column in user_matrix.
        users_col = self.user_matrix.col(self.user_ids_dict[user_id])
        users_col = users_col.transpose
        # Multiply user column with item_matrix.
        items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
        # Get the item_id by column index.
        items_scores = map(lambda x: (self.item_ids[x[0]], x[1]), items_col)
        # Filter the item which user has already viewed.
        viewed_items = self.user_items[user_id]
        items_scores = filter(lambda x: x[0] not in viewed_items, items_scores)
        # Get the top n_items by item score.
        return sorted(items_scores, key=lambda x: x[1], reverse=True)[:n_items]

    def predict(self, user_ids, n_items=10):
        """Predict the items ratings by users.

        Arguments:
            user_ids {list} -- 1d list with int.

        Keyword Arguments:
            n_items {int} -- Number of items. (default: {10})

        Returns:
            list -- 2d list with item_id and score.
        """

        return [self._predict(user_id, n_items) for user_id in user_ids]
