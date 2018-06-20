# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-06-13 14:55:11 
@Last Modified by: tushushu 
@Last Modified time: 2018-06-13 14:55:11 
"""

from math import log2
from collections import namedtuple
from copy import copy


class Node(object):
    def __init__(self, idx=None, prob=None):
        """Node class to build tree leaves.

        Keyword Arguments:
            idx {list} -- 1d list object with int (default: {None})
            prob {float} -- positive probability (default: {None})
        """
        self.idx = idx
        self.prob = prob

        self.left = None
        self.right = None
        self.feature = None
        self.split = None


class DecisionTree(object):
    def __init__(self):
        """DecisionTree class only support binary classification with ID3.
        Attributes:
        root: the root node of DecisionTree
        height: the height of DecisionTree
        _split_effect: number of samples, positive probability and rate in split set of each split group
        """

        self.root = Node()
        self.height = 0
        self._split_effect = namedtuple(
            "split_effect", ["idx", "cnt", "prob", "rate"])

    def _get_split_effect(self, X, y, idx, feature, split):
        """List length, positive probability and rate when x is splitted into two pieces.

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- 1d list object with int
            feature {int} -- Feature number
            split {float} -- Split point of x

        Returns:
            named tuple -- split effect
        """

        n = len(idx)
        pos_cnt = [0, 0]
        cnt = [0, 0]
        idx_split = [[], []]
        # Iterate each row and compare with the split point
        for i in idx:
            xi, yi = X[i][feature], y[i]
            if xi < split:
                cnt[0] += 1
                pos_cnt[0] += yi
            else:
                cnt[1] += 1
                pos_cnt[1] += yi
        # Calculate the split effect
        prob = [pos_cnt[0] / cnt[0], pos_cnt[1] / cnt[1]]
        rate = [cnt[0] / n, cnt[1] / n]
        return self._split_effect(idx_split, cnt, prob, rate)

    def _get_entropy(self, p):
        """Calculate entropy

        Arguments:
            p {float} -- Positive probability

        Returns:
            float -- Entropy
        """
        # According to L'Hospital's rule, 0 * log2(0) equals to zero.
        if p == 1 or p == 0:
            return 0
        else:
            q = 1 - p
            return -(p * log2(p) + q * log2(q))

    def _get_info(self, y, idx):
        """Calculate info of y

        Arguments:
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- 1d list object with int

        Returns:
            float -- Info of y
        """

        p = sum([y[i] for i in idx]) / len(idx)
        return self._get_entropy(p)

    def _get_cond_info(self, se):
        """Calculate conditonal info of x

        Arguments:
            se {named tuple} -- Split effect

        Returns:
            float -- Conditonal info of x
        """

        info_left = self._get_entropy(se.prob[0])
        info_right = self._get_entropy(se.prob[1])
        return se.rate[0] * info_left + se.rate[1] * info_right

    def _choose_split_point(self, X, y, idx, feature):
        """Iterate each xi and split x, y into two pieces,
        and the best split point is the xi when we get max gain.

        Arguments:
            x {list} -- 1d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- 1d list object with int
            feature {int} -- Feature number

        Returns:
            tuple -- The best choice of feature, split and split effect
        """
        # Feature cannot be splitted if there's only one unique element.
        unique = set([X[i][feature] for i in idx])
        if len(unique) == 1:
            return None
        # In case of empty split
        unique.remove(min(unique))

        def f(split):
            """Auxiliary function of _choose_split_poin
            """

            info = self._get_info(y, idx)
            se = self._get_split_effect(X, y, idx, feature, split)
            cond_info = self._get_cond_info(se)
            gain = info - cond_info
            return gain, split, se

        # Get split point which has max gain
        gain, split, se = max((f(split)
                               for split in unique), key=lambda x: x[0])
        return gain, feature, split, se

    def _choose_feature(self, X, y, idx):
        """Choose the feature which has max info gain.

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- 1d list object with int

        Returns:
            tuple -- (feature number, split point, split effect)
        """

        m = len(X[0])
        # Compare the info gain of each feature and choose best one.
        split_rets = [x for x in map(lambda x: self._choose_split_point(
            X, y, idx, x), range(m)) if x is not None]
        # Terminate if no feature can be splitted
        if split_rets == []:
            return None
        _, feature, split, se = max(split_rets, key=lambda x: x[0])
        # Get split idx into two pieces and empty idx
        while idx:
            i = idx.pop()
            xi = X[i][feature]
            if xi < split:
                se.idx[0].append(i)
            else:
                se.idx[1].append(i)
        return feature, split, se

    def _expr2literal(self, expr):
        """Auxiliary function of print_rules.

        Arguments:
            expr {list} -- 1D list like [Feature, op, split]

        Returns:
            str
        """

        feature, op, split = expr
        op = ">=" if op == 1 else "<"
        return "Feature%d %s %.4f" % (feature, op, split)

    def _get_rules(self):
        """Get the rules of all the decision tree leaf nodes. 
            Expr: 1D list like [Feature, op, split]
            Rule: 2D list like [[Feature, op, split], prob]
            Op: -1 means less than, 1 means equal or more than
        """

        que = [[self.root, []]]
        self.rules = []
        # Breadth-First Search
        while que:
            nd, exprs = que.pop(0)
            # Generate a rule when the current node is leaf node
            if not(nd.left or nd.right):
                # Convert expression to text
                literals = list(map(self._expr2literal, exprs))
                self.rules.append([literals, nd.prob])
            # Expand when the current node has left child
            if nd.left:
                rule_left = copy(exprs)
                rule_left.append([nd.feature, -1, nd.split])
                que.append([nd.left, rule_left])
            # Expand when the current node has right child
            if nd.right:
                rule_right = copy(exprs)
                rule_right.append([nd.feature, 1, nd.split])
                que.append([nd.right, rule_right])

    def fit(self, X, y, max_depth=3, min_samples_split=2):
        """Build a decision tree classifier.
        Note:
            At least there's one column in X has more than 2 unique elements
            y cannot be all 1 or all 0

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1

        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {2})
            min_samples_split {int} -- The minimum number of samples required to split an internal node (default: {2})
        """

        # Initialize with depth, node
        self.root = Node(list(range(len(y))))
        que = [[0, self.root]]
        # Breadth-First Search
        while que:
            depth, nd = que.pop(0)
            # Terminate loop if tree depth is more than max_depth
            if depth == max_depth:
                break
            # Stop split when number of node samples is less than min_samples_split or Node is 100% pure.
            if len(nd.idx) < min_samples_split or nd.prob == 1 or nd.prob == 0:
                continue
            # Stop split if no feature has more than 2 unique elements
            feature_rets = self._choose_feature(X, y, nd.idx)
            if feature_rets is None:
                continue
            # Split
            nd.feature, nd.split, se = feature_rets
            nd.left = Node(se.idx[0], se.prob[0])
            nd.right = Node(se.idx[1], se.prob[1])
            que.append([depth+1, nd.left])
            que.append([depth+1, nd.right])
        # Update tree depth and rules
        self.height = depth
        self._get_rules()

    def print_rules(self):
        """Print the rules of all the decision tree leaf nodes.
        """

        for i, rule in enumerate(self.rules):
            literals, prob = rule
            print("Rule %d: " % i, ' | '.join(
                literals) + ' => Prob %.4f' % prob)

    def _predict_prob(self, row):
        """Auxiliary function of predict_prob.

        Arguments:
            row {list} -- 1D list with int or float

        Returns:
            float
        """

        nd = self.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd.prob

    def predict_prob(self, X):
        """Get the probability that y is positive.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        return [self._predict_prob(row) for row in X]

    def predict(self, X, threshold=0.5):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Keyword Arguments:
            threshold {float} -- Prediction = 1 when probability >= threshold (default: {0.5})

        Returns:
            list -- 1d list object with float
        """

        return [int(y >= threshold) for y in self.predict_prob(X)]


if __name__ == "__main__":
    from time import time
    from utils import load_breast_cancer, train_test_split

    start = time()
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)
    # Train model
    clf = DecisionTree()
    clf.fit(X_train, y_train)
    # Show rules
    clf.print_rules()
    # Model accuracy
    acc = sum((y_test_hat == y_test for y_test_hat, y_test in zip(
        clf.predict(X_test), y_test))) / len(y_test)
    print("Test accuracy is %.2f%%!" % (acc * 100))
    # Show run time, you can try it in Pypy which might be 10x faster.
    print("Total run time is %.2f s" % (time() - start))
