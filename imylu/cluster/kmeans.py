# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-01 14:31:13
@Last Modified by:   tushushu
@Last Modified time: 2018-11-01 14:31:13
"""
from ..utils import get_euclidean_distance, get_cosine_distance
from random import random
from collections import Counter


class KMeans(object):
    def __init__(self):
        self.k = None
        self.n_features = None
        self.cluster_centers = None
        self.distance_fn = None
        self.cluster_samples_cnt = None

    def _cmp_arr(self, arr1, arr2, tolerance=1e-5):
        """Compare if the elements of two lists are the same.

        Arguments:
            arr1 {list} -- 1d list with int or float.
            arr2 {list} -- 1d list with int or float.

        Keyword Arguments:
            tolerance {float} -- The minimum acceptable difference.
            (default: {1e-5})

        Returns:
            bool
        """

        return all(abs(a - b) < tolerance for a, b in zip(arr1, arr2))

    def _init_cluster_centers(self, k):
        """Generate initial cluster centers with empty list.

        Arguments:
            k {int} -- Number of cluster centers.

        Returns:
            list
        """

        return [[] for _ in range(k)]

    def _get_cluster_centers(self, X, k, n_features, cluster_nums,
                             cluster_samples_cnt):
        """Calculate the cluster centers by the average of each cluster's samples.

        Arguments:
            X {list} -- 2d list with int or float.
            k {int} -- Number of cluster centers.
            n_features {int} -- Number of features.
            cluster_nums {list} -- 1d list with int.
            cluster_samples_cnt{Counter} -- Count of samples in each cluster.

        Returns:
            list -- 2d list with int or float.
        """

        ret = [[0 for _ in range(n_features)] for _ in range(k)]
        for Xi, cetner_num in zip(X, cluster_nums):
            ret[cetner_num] += Xi / cluster_samples_cnt[cetner_num]
        return ret

    def _get_nearest_center(self, Xi, centers, distance_fn):
        """Search the nearest cluster center of Xi.

        Arguments:
            Xi {list} -- 1d list with int or float.
            centers {list} -- 2d list with int or float.
            distance_fn {function} -- The function to measure the distance.

        Returns:
            int -- Cluster center number.
        """

        return min(((i, distance_fn(Xi, center))
                    for i, center in enumerate(centers)), key=lambda x: x[1])

    def _get_nearest_centers(self, X, centers, distance_fn):
        """Search the nearest cluster centers of X.

        Arguments:
            X {list} -- 2d list with int or float.
            centers {list} -- 2d list with int or float.
            distance_fn {function} -- The function to measure the distance.

        Returns:
            list
        """

        return [self._get_nearest_center(Xi, centers, distance_fn) for Xi in X]

    def _get_empty_cluster_nums(self, cluster_samples_cnt):
        """Filter empty cluster numbers.

        Arguments:
            cluster_samples_cnt {Counter} -- Count of samples in each cluster.

        Returns:
            list
        """

        empty_clusters = filter(
            lambda x: x[1] == 0, cluster_samples_cnt.items())
        return [empty_cluster[0] for empty_cluster in empty_clusters]

    def _process_empty_clusters(self, centers, empty_cluster_nums):
        """Replace empty clusters with new clusters centers.

        Arguments:
            centers {list} -- 2d list with int or float.
            empty_cluster_nums {list} -- 1d list with int.

        Returns:
            list
        """
        n_features = len(centers[0])
        for i in empty_cluster_nums:
            center_cur = []
            # In case of duplicate cluster centers.
            while any(self._cmp_arr(center_cur, center) for center in centers):
                center_cur = [random() for _ in range(n_features)]
            centers[i] = center_cur
        return centers

    def _has_empty_cluster(self, k, cluster_samples_cnt):
        """If empty cluster exists.

        Arguments:
            k {int} -- Number of cluster centers.
            cluster_samples_cnt{Counter} -- Count of samples in each cluster.

        Returns:
            bool
        """

        return any(map(lambda i: cluster_samples_cnt[i] == 0, range(k)))

    def _is_converged(self, centers, centers_new):
        """If the algorithm converges.

        Arguments:
            centers {list} -- 2d list with int or float.
            centers_new {list} -- 2d list with int or float.

        Returns:
            bool
        """

        return all(self._cmp_arr(arr1, arr2) for arr1, arr2
                   in zip(centers, centers_new))

    def fit(self, X, k, fn=None, n_iter=100):
        """Build K-Means model.

        Arguments:
            X {list} -- 2d list with int or float.
            k {int} -- Number of cluster centers.

        Keyword Arguments:
            fn {str} -- The function to measure the distance.
            (default: {None})
            n_iter {int} -- Number of iterations. (default: {100})
        """

        # Number of features.
        n_features = len(X[0])

        # Distance functions for rows.
        if fn is None:
            distance_fn = get_euclidean_distance
        else:
            error_msg = "Parameter distance_fn must be eu or cos!"
            assert fn in ("eu", "cos"), error_msg
            if fn == "eu":
                distance_fn = get_euclidean_distance
            if fn == "cos":
                distance_fn = get_cosine_distance

        # Calculate cluster centers.
        # Initialization.
        centers = self._init_cluster_centers(k, n_features)
        cluster_samples_cnt = Counter()
        for _ in range(n_iter):
            # No empty clusters.
            while self._has_empty_cluster(k, cluster_samples_cnt):
                # Empty cluster numbers.
                empty_cluster_nums = self._get_empty_cluster_nums(
                    cluster_samples_cnt)
                # Process empty clusters.
                centers = self._process_empty_clusters(
                    centers, empty_cluster_nums)
                # Search the nearest cluster centers of X
                cluster_nums = self._get_nearest_centers(
                    X, centers, distance_fn)
                # Count of samples in each cluster.
                cluster_samples_cnt = Counter(cluster_nums)

            # New cluster centers.
            centers_new = self._get_cluster_centers(
                X, k, n_features, cluster_nums, cluster_samples_cnt)
            # If the algorithm converges.
            if self._is_converged(centers, centers_new):
                break
            # Update current cluster centers.
            centers = centers_new

        # The properties of K-Means model.
        self.k = k
        self.n_features = n_features
        self.distance_fn = distance_fn
        self.cluster_centers = centers
        self.cluster_samples_cnt = cluster_samples_cnt

    def _predict(self, Xi):
        """Get the cluster center of Xi.

        Arguments:
            Xi {list} -- 1d list with int or float.

        Returns:
            int -- cluster center
        """

        return self._get_nearest_center(Xi, self.cluster_centers,
                                        self.distance_fn)

    def predict(self, X):
        """Get the cluster center of X.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with int or float
        """

        return [self._predict(Xi) for Xi in X]
