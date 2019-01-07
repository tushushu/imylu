# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-10-15 19:46:58
@Last Modified by:   tushushu
@Last Modified time: 2018-10-15 19:46:58
"""
from itertools import product, chain
from copy import deepcopy


class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def row(self, row_no):
        """Get a row of the matrix.

        Arguments:
            row_no {int} -- Row number of the matrix.

        Returns:
            Matrix
        """

        return Matrix([self.data[row_no]])

    def col(self, col_no):
        """Get a column of the matrix.

        Arguments:
            col_no {int} -- Column number of the matrix.

        Returns:
            Matrix
        """
        m = self.shape[0]
        return Matrix([[self.data[i][col_no]] for i in range(m)])

    @property
    def is_square(self):
        """Check if the matrix is a square matrix.

        Returns:
            bool
        """

        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        """Find the transpose of the original matrix.

        Returns:
            Matrix
        """

        data = list(map(list, zip(*self.data)))
        return Matrix(data)

    def _eye(self, n):
        """Get a unit matrix with shape (n, n).

        Arguments:
            n {int} -- Rank of unit matrix.

        Returns:
            list
        """

        return [[0 if i != j else 1 for j in range(n)] for i in range(n)]

    @property
    def eye(self):
        """Get a unit matrix with the same shape of self.

        Returns:
            Matrix
        """

        assert self.is_square, "The matrix has to be square!"
        data = self._eye(self.shape[0])
        return Matrix(data)

    def _gaussian_elimination(self, aug_matrix):
        """To simplify the left square matrix of the augmented matrix
        as a unit diagonal matrix.

        Arguments:
            aug_matrix {list} -- 2d list with int or float.

        Returns:
            list -- 2d list with int or float.
        """

        n = len(aug_matrix)
        m = len(aug_matrix[0])

        # From top to bottom.
        for col_idx in range(n):
            # Check if element on the diagonal is zero.
            if aug_matrix[col_idx][col_idx] == 0:
                row_idx = col_idx
                # Find a row whose element has same column index with
                # the element on the diagonal is not zero.
                while row_idx < n and aug_matrix[row_idx][col_idx] == 0:
                    row_idx += 1
                # Add this row to the row of the element on the diagonal.
                for i in range(col_idx, m):
                    aug_matrix[col_idx][i] += aug_matrix[row_idx][i]

            # Elimiate the non-zero element.
            for i in range(col_idx + 1, n):
                # Skip the zero element.
                if aug_matrix[i][col_idx] == 0:
                    continue
                # Elimiate the non-zero element.
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in range(col_idx, m):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # From bottom to top.
        for col_idx in range(n - 1, -1, -1):
            # Elimiate the non-zero element.
            for i in range(col_idx):
                # Skip the zero element.
                if aug_matrix[i][col_idx] == 0:
                    continue
                # Elimiate the non-zero element.
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in chain(range(i, col_idx + 1), range(n, m)):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # Iterate the element on the diagonal.
        for i in range(n):
            k = 1 / aug_matrix[i][i]
            aug_matrix[i][i] *= k
            for j in range(n, m):
                aug_matrix[i][j] *= k

        return aug_matrix

    def _inverse(self, data):
        """Find the inverse of a matrix.

        Arguments:
            data {list} -- 2d list with int or float.

        Returns:
            list -- 2d list with int or float.
        """

        n = len(data)
        unit_matrix = self._eye(n)
        aug_matrix = [a + b for a, b in zip(self.data, unit_matrix)]
        ret = self._gaussian_elimination(aug_matrix)
        return list(map(lambda x: x[n:], ret))

    @property
    def inverse(self):
        """Find the inverse matrix of self.

        Returns:
            Matrix
        """

        assert self.is_square, "The matrix has to be square!"
        data = self._inverse(self.data)
        return Matrix(data)

    def _row_mul(self, row_A, row_B):
        """Multiply the elements with the same subscript in both arrays and sum them.

        Arguments:
            row_A {list} -- 1d list with float or int.
            row_B {list} -- 1d list with float or int.

        Returns:
            float or int
        """

        return sum(x[0] * x[1] for x in zip(row_A, row_B))

    def _mat_mul(self, row_A, B):
        """An auxiliary function of the mat_mul function.

        Arguments:
            row_A {list} -- 1d list with float or int.
            B {Matrix}

        Returns:
            list -- 1d list with float or int.
        """

        row_pairs = product([row_A], B.transpose.data)
        return [self._row_mul(*row_pair) for row_pair in row_pairs]

    def mat_mul(self, B):
        """Matrix multiplication.

        Arguments:
            B {Matrix}

        Returns:
            Matrix
        """

        error_msg = "A's column count does not match B's row count!"
        assert self.shape[1] == B.shape[0], error_msg
        return Matrix([self._mat_mul(row_A, B) for row_A in self.data])

    def _mean(self, data):
        """Calculate the average of all the samples.

        Arguments:
            X {list} -- 2d list with int or float.

        Returns:
            list -- 1d list with int or float.
        """

        m = len(data)
        n = len(data[0])
        ret = [0 for _ in range(n)]
        for row in data:
            for j in range(n):
                ret[j] += row[j] / m
        return ret

    def mean(self):
        """Calculate the average of all the samples.

        Returns:
            Matrix
        """

        return Matrix(self._mean(self.data))

    def scala_mul(self, scala):
        """Scala multiplication.

        Arguments:
            scala {float}

        Returns:
            Matrix
        """

        m, n = self.shape
        data = deepcopy(self.data)
        for i in range(m):
            for j in range(n):
                data[i][j] *= scala
        return Matrix(data)
