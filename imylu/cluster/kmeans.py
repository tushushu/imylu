# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-01 14:31:13
@Last Modified by:   tushushu
@Last Modified time: 2018-11-01 14:31:13
"""
from ..utils import get_euclidean_distance, get_cosine_distance
from random import random, randint
from collections import Counter
from copy import deepcopy


class KMeans(object):
    def __init__(self):
        self.k = None
        self.n_features = None
        self.cluster_centers = None
        self.distance_fn = None
        self.cluster_samples_cnt = None

    def _bin_search(self, target, nums):
        """Binary search a number from array-like object.

        Arguments:
            target {float}
            nums {list}

        Returns:
            int
        """

        low = 0
        high = len(nums) - 1
        assert nums[low] <= target < nums[high], "Cannot find target!"
        while 1:
            mid = (low + high) // 2
            if mid == 1 or target >= nums[mid]:
                low = mid + 1
            elif target < nums[mid - 1]:
                high = mid - 1
            else:
                break
        return mid

    def _init_cluster_centers(self, X, k, n_features, distance_fn):
        """Generate initial cluster centers with K-means++.

        Arguments:
            X {list} -- 2d list with int or float.
            k {int} -- Number of cluster centers.
            n_features {int} -- Number of features.

        Returns:
            list
        """

        n = len(X)
        centers = [X[randint(0, n - 1)]]
        for _ in range(k - 1):
            center_pre = centers[-1]
            # Calculate distances of center_pre to all the rows in X.
            indexes_dists = ([i, distance_fn(Xi, center_pre)]
                             for i, Xi in enumerate(X))
            # Sort the distances.
            indexes_dists = sorted(indexes_dists, key=lambda x: x[1])
            # Get distances.
            dists = [x[1] for x in indexes_dists]
            # Scale.
            tot = sum(dists)
            dists = [dist / tot for dist in dists]
            # Cumsum.
            for i in range(1, n):
                dists[i] += dists[i - 1]
            # In case of duplicate cluster centers.
            while 1:
                num = random()
                i = self._bin_search(num, dists)
                center_cur = X[i]
                if not any(self._cmp_arr(center_cur, center)
                           for center in centers):
                    break
            centers.append(center_cur)
        return centers

    def _cmp_arr(self, arr1, arr2, tolerance=1e-8):
        """Compare if the elements of two lists are the same.

        Arguments:
            arr1 {list} -- 1d list with int or float.
            arr2 {list} -- 1d list with int or float.

        Keyword Arguments:
            tolerance {float} -- The minimum acceptable difference.
            (default: {1e-8})

        Returns:
            bool
        """

        return len(arr1) == len(arr2) and \
            all(abs(a - b) < tolerance for a, b in zip(arr1, arr2))

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
            for j in range(n_features):
                ret[cetner_num][j] += Xi[j] / cluster_samples_cnt[cetner_num]
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

        return min(((i, distance_fn(Xi, center)) for i, center
                    in enumerate(centers)), key=lambda x: x[1])[0]

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

    def _get_empty_cluster_nums(self, cluster_samples_cnt, k):
        """Filter empty cluster numbers.

        Arguments:
            cluster_samples_cnt {Counter} -- Count of samples in each cluster.
            k {int} -- Number of cluster centers.

        Returns:
            list
        """

        clusters = ((i, cluster_samples_cnt[i]) for i in range(k))
        empty_clusters = filter(lambda x: x[1] == 0, clusters)
        return [empty_cluster[0] for empty_cluster in empty_clusters]

    def _process_empty_clusters(self, centers, empty_cluster_nums, n_features):
        """Replace empty clusters with new clusters centers.

        Arguments:
            centers {list} -- 2d list with int or float.
            empty_cluster_nums {list} -- 1d list with int.
            n_features {int} -- Number of features.

        Returns:
            list
        """

        for i in empty_cluster_nums:
            center_cur = [random() for _ in range(n_features)]
            # In case of duplicate cluster centers.
            while any(self._cmp_arr(center_cur, center) for center in centers):
                center_cur = [random() for _ in range(n_features)]
            centers[i] = center_cur
        return centers

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
        centers = self._init_cluster_centers(X, k, n_features, distance_fn)
        for i in range(n_iter):
            while 1:
                # Search the nearest cluster centers of X
                cluster_nums = self._get_nearest_centers(
                    X, centers, distance_fn)
                # Count of samples in each cluster.
                cluster_samples_cnt = Counter(cluster_nums)
                # Empty cluster numbers.
                empty_cluster_nums = self._get_empty_cluster_nums(
                    cluster_samples_cnt, k)
                # No empty clusters.
                if empty_cluster_nums:
                    # Process empty clusters.
                    centers = self._process_empty_clusters(
                        centers, empty_cluster_nums, n_features)
                else:
                    break

            # New cluster centers.
            centers_new = self._get_cluster_centers(
                X, k, n_features, cluster_nums, cluster_samples_cnt)
            # If the algorithm converges.
            if self._is_converged(centers, centers_new):
                break
            # Update current cluster centers.
            centers = deepcopy(centers_new)

        print("Iteration: %d" % i)
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
