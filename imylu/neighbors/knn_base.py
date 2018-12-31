# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-13 17:15:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-13 17:15:29
"""
from ..utils.kd_tree import KDTree
from ..utils.max_heap import MaxHeap


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
        nd = tree._search(Xi, tree.root)
        # Initialize a queue.
        que = [(tree.root, nd)]
        while que:
            nd_root, nd_cur = que.pop(0)
            # Calculate distance between Xi and root node
            nd_root.dist = tree._get_eu_dist(Xi, nd_root)
            # Update best node and distance.
            heap.add(nd_root)
            while nd_cur is not nd_root:
                # Calculate distance between Xi and current node
                nd_cur.dist = tree._get_eu_dist(Xi, nd_cur)
                # Update best node and distance
                heap.add(nd_cur)
                # If it's necessary to visit brother node.
                if nd_cur.brother and \
                        (not heap or
                         heap.items[0].dist >
                         tree._get_hyper_plane_dist(Xi, nd_cur.father)):
                    _nd = tree._search(Xi, nd_cur.brother)
                    que.append((nd_cur.brother, _nd))
                # Back track.
                nd_cur = nd_cur.father
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
            list -- 1d list object with int or float
        """

        return [self._predict(Xi) for Xi in X]
