# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-13 17:15:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-13 17:15:29
"""
from .kd_tree import KDTree
from .max_heap import MaxHeap


class KNeighborsBase(object):
    """KNN base class.

    Attributes:
        k_neighbors {int}: Learning rate.
        trees {list}: 1d list with RegressionTree objects.
    """

    def __init__(self):
        self.k_neighbors = None
        self.tree = None

    def fit(self, X, y, k_neighbors=3):
        """Build KNN model.

        Arguments:
            X {list} -- 2d list with int or float.
            y {list} -- 1d list object with int or float.

        Keyword Arguments:
            k_neighbors {int} -- Number of neighbors to search. (default: {3})
        """

        self.k_neighbors = k_neighbors
        self.tree = KDTree()
        self.tree.build_tree(X, y)

    def _knn_search(self, Xi):
        """K nearest neighbours search and backtracking.

        Arguments:
            Xi {list} -- 1d list with int or float.

        Returns:
            list -- K nearest nodes to Xi.
        """

        tree = self.tree
        heap = MaxHeap(self.k_neighbors, lambda x: x.dist)
        # The path from root to a leaf node when searching Xi.
        path = tree._search(Xi, tree.root)
        # Record which nodes' brohters has been visited already.
        bro_flags = [1] + [0] * (len(path) - 1)
        que = list(zip(path, bro_flags))
        while 1:
            nd_cur, bro_flag = que.pop()
            # Calculate distance between Xi and current node
            nd_cur.dist = tree._get_eu_dist(Xi, nd_cur)
            # Update best node and distance
            heap.add(nd_cur)
            # Calculate distance between Xi and father node's hyper plane.
            if que:
                nd_dad = que[-1][0]
            else:
                break
            # If it's necessary to visit brother node.
            nd_bro = tree._get_brother(nd_cur, nd_dad)
            if (bro_flag == 1 or nd_bro is None) and \
                    heap.size == heap.max_size:
                continue
            # Check if it's possible that the other side of father node
            # has closer child node.
            dist_hyper = tree._get_hyper_plane_dist(Xi, nd_dad)
            if nd_cur.dist > dist_hyper:
                _path = tree._search(Xi, nd_bro)
                _bro_flags = [1] + [0] * (len(_path) - 1)
                que.extend(zip(_path, _bro_flags))
            else:
                continue
        return heap

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1D list with int or float

        Returns:
            int or float -- prediction of yi
        """

        return NotImplemented

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            NotImplemented
        """

        return NotImplemented
