# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-01 14:31:13
@Last Modified by:   tushushu
@Last Modified time: 2018-11-01 14:31:13
"""

from collections import Counter
from copy import deepcopy
from random import random, randint
from ..utils.utils import get_euclidean_distance, get_cosine_distance


class KMeans(object):
    """KMeans class.

    Attributes:
        k {int} -- Number of cluster centers.
        n_features {int} -- Number of features.
        cluster_centers {list} -- 2d list with int or float.
        distance_fn {function} -- The function to measure the distance.
        cluster_samples_cnt {Counter} --  Count of samples in each cluster.
    """

    def __init__(self):
        self.k = None
        self.n_features = None
        self.cluster_centers = None
        self.distance_fn = None
        self.cluster_samples_cnt = None

    def _bin_search(self, target, nums):
        """Binary search a number from array-like object.
        The result is the minimum number greater than target in nums.

        Arguments:
            target {float}
            nums {list}

        Returns:
            int -- The index result in nums.
        """

        low = 0
        high = len(nums) - 1
        assert nums[low] <= target < nums[high], "Cannot find target!"
        while 1:
            mid = (low + high) // 2
            if mid == 0 or target >= nums[mid]:
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
            distance_fn {function} -- The function to measure the distance.

        Returns:
            list
        """

        n = len(X)
        centers = [X[randint(0, n - 1)]]
        for _ in range(k - 1):
            center_pre = centers[-1]
            # Get the indexes and distances of center_pre to all the rows in X.
            idxs_dists = ([i, distance_fn(Xi, center_pre)]
                          for i, Xi in enumerate(X))
            # Sort the indexes and distance pair by distance.
            idxs_dists = sorted(idxs_dists, key=lambda x: x[1])
            # Get the distances for index dist pairs.
            dists = [x[1] for x in idxs_dists]
            # Get the summary of distances.
            tot = sum(dists)
            # Scale the distances.
            for i in range(1, n):
                dists[i] /= tot
            # Cumulative sum the distances.
            for i in range(1, n):
                dists[i] += dists[i - 1]
            # Randomly choose a row in X as cluster center.
            while 1:
                num = random()
                # Search the minimum distances greater than num.
                dist_idx = self._bin_search(num, dists)
                row_idx = idxs_dists[dist_idx][0]
                # Set the corresponding row to the distance we searched
                # as the new cluster center.
                center_cur = X[row_idx]
                # In case of duplicate cluster centers.
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

    def _get_cluster_centers(self, X, k, n_features, y, cluster_samples_cnt):
        """Calculate the cluster centers by the average of each cluster's samples.

        Arguments:
            X {list} -- 2d list with int or float.
            k {int} -- Number of cluster centers.
            n_features {int} -- Number of features.
            y {list} -- 1d list with int.
            cluster_samples_cnt{Counter} -- Count of samples in each cluster.

        Returns:
            list -- 2d list with int or float.
        """

        ret = [[0 for _ in range(n_features)] for _ in range(k)]
        for Xi, cetner_num in zip(X, y):
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

    def _get_nearest_centers(self, X, distance_fn, centers):
        """Search the nearest cluster centers of X.

        Arguments:
            X {list} -- 2d list with int or float.
            distance_fn {function} -- The function to measure the distance.
            centers {list} -- 2d list with int or float.

        Returns:
            list
        """

        return [self._get_nearest_center(Xi, centers, distance_fn) for Xi in X]

    def _get_empty_cluster_idxs(self, cluster_samples_cnt, k):
        """Filter the index of empty cluster.

        Arguments:
            cluster_samples_cnt {Counter} -- Count of samples in each cluster.
            k {int} -- Number of cluster centers.

        Returns:
            list
        """

        clusters = ((i, cluster_samples_cnt[i]) for i in range(k))
        empty_clusters = filter(lambda x: x[1] == 0, clusters)
        return [empty_cluster[0] for empty_cluster in empty_clusters]

    def _get_furthest_row(self, X, distance_fn, centers,
                          empty_cluster_idxs):
        """Find the row in X which is furthest to all the non empty centers.

        Arguments:
            X {list} -- 2d list with int or float.
            distance_fn {function} -- The function to measure the distance.
            centers {list} -- 2d list with int or float.
            empty_cluster_idxs {list} -- Non empty cluster centers' indexes.

        Returns:
            list -- 1d list with int or float.
        """

        # Function to calculate the distance of Xi to each cluster centers.
        def f(Xi, centers):
            return sum(distance_fn(Xi, centers) for center in centers)
        # Filter the non empty cluster centers.
        non_empty_centers = map(lambda x: x[1], filter(
            lambda x: x[0] not in empty_cluster_idxs, enumerate(centers)))
        # Find the row in X which is furthest to all the non empty centers.
        return max(map(lambda x: [x, f(x, non_empty_centers)], X),
                   key=lambda x: x[1])[0]

    def _process_empty_clusters(self, X, distance_fn, n_features, centers,
                                empty_cluster_idxs):
        """Replace empty clusters with new clusters centers.

        Arguments:
            X {list} -- 2d list with int or float.
            distance_fn {function} -- The function to measure the distance.
            n_features {int} -- Number of features.
            centers {list} -- 2d list with int or float.
            empty_cluster_nums {list} -- 1d list with int.

        Returns:
            list
        """

        for i in empty_cluster_idxs:
            center_cur = self._get_furthest_row(X, distance_fn, centers,
                                                empty_cluster_idxs)
            # In case of duplicate cluster centers.
            while any(self._cmp_arr(center_cur, center) for center in centers):
                center_cur = self._get_furthest_row(X, distance_fn, centers,
                                                    empty_cluster_idxs)
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
                y = self._get_nearest_centers(
                    X, distance_fn, centers)
                # Count of samples in each cluster.
                cluster_samples_cnt = Counter(y)
                # Empty cluster numbers.
                empty_cluster_idxs = self._get_empty_cluster_idxs(
                    cluster_samples_cnt, k)
                # No empty clusters.
                if empty_cluster_idxs:
                    # Process empty clusters.
                    centers = self._process_empty_clusters(
                        centers, empty_cluster_idxs, n_features)
                else:
                    break

            # New cluster centers.
            centers_new = self._get_cluster_centers(
                X, k, n_features, y, cluster_samples_cnt)
            # If the algorithm converges.
            if self._is_converged(centers, centers_new):
                break
            # Update current cluster centers.
            centers = deepcopy(centers_new)

        print("Iterations: %d" % i)
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
