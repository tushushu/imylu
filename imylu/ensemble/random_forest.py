# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-26 14:41:08
@Last Modified by: tushushu
@Last Modified time: 2019-05-28 19:41:08
"""

from numpy import array, mean, apply_along_axis
from numpy.random import choice, seed
from ..tree.decision_tree import DecisionTree, Node


class RandomTree(DecisionTree):
    """RandomTree class.

    Attributes:
        root {Node} -- Root node of DecisionTree.
        depth {int} -- Depth of DecisionTree.
        _rules {list} -- Rules of all the tree nodes.
        max_features {int} -- Number of features when split.
    """

    def __init__(self, max_features: int):
        super(RandomTree, self).__init__()
        self.max_features = max_features

    def _choose_feature(self, data: array, label: array) -> Node:
        """Choose the feature which has maximum info gain randomly.

        Arguments:
            data {array} -- Training data.
            label {array} -- Target values.

        Returns:
            Node -- feature number, split point, prob.
        """

        # Choose max_features features randomly without replacement.
        features = choice(
            range(data.shape[1]), size=self.max_features, replace=False)
        # Compare the mse of each feature and choose best one.
        _ite = map(lambda x: (self._choose_split(
            data[:, x], label), x), features)
        ite = filter(lambda x: x[0].split is not None, _ite)

        # Return None if no feature can be splitted.
        node, feature = max(
            ite, key=lambda x: x[0].gain, default=(Node(), None))
        node.feature = feature

        return node


class RandomForest:
    """RandomForest, randomly build some DecisionTree instance,
    and the average score of each DecisionTree.

    Attributes:
    trees {list} -- 1d list with DecisionTree objects
    """

    def __init__(self):
        self.trees = None

    def fit(self, data: array, label: array, n_estimators=10, max_depth=3, min_samples_split=2,
            max_features=None, random_state=None):
        """Build a RandomForest classifier.

        Arguments:
            data {array} -- Training data.
            label {array} -- Target values.

        Keyword Arguments:
            n_estimators {int} -- Number of trees. (default: {5})
            max_depth {int} -- Maximum depth of each tree. (default: {3})
            min_samples_split {int} -- Minimum number of samples required
            to split an internal node. (default: {2})
            max_features {int} -- Number of features when split. (default: {None})
            random_state {int} -- The seed used by the random number generator. (default: {None})
        """

        # Set random state.
        if random_state is not None:
            seed(random_state)

        self.trees = []
        for _ in range(n_estimators):
            n_cols, n_rows = data.shape

            # Choose rows randomly with replacement.
            idx = choice(range(n_rows), size=n_rows, replace=True)

            # Choose columns randomly without replacement.
            if max_features:
                max_features = min(n_cols, max_features)
            else:
                max_features = int(n_cols ** 0.5)

            # Subsample of data and label.
            data_sub = data[idx, :]
            label_sub = label[idx]

            # Train decision tree classifier
            clf = RandomTree(max_features)
            clf.fit(data_sub, label_sub, max_depth, min_samples_split)
            self.trees.append(clf)
        
        # Cancel random state.
        if random_state is not None:
            seed(None)

    def predict_one_prob(self, row: array) -> float:
        """Auxiliary function of predict_prob.

        Arguments:
            row {array} -- A sample of testing data.

        Returns:
            float -- Prediction of label.
        """

        return mean([tree.predict_one_prob(row) for tree in self.trees])

    def predict_prob(self, data: array) -> array:
        """Get the probability of label.

        Arguments:
            data {array} -- Testing data.

        Returns:
            array -- Probabilities of label.
        """

        return apply_along_axis(self.predict_one_prob, axis=1, arr=data)

    def predict(self, data: array, threshold=0.5):
        """Get the prediction of label.

        Arguments:
            data {array} -- Testing data.

        Keyword Arguments:
            threshold {float} -- (default: {0.5})

        Returns:
            array -- Prediction of label.
        """

        prob = self.predict_prob(data)
        return (prob >= threshold).astype(int)
