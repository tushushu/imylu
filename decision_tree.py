from math import log2
from collections import namedtuple
from copy import copy, deepcopy


class Node(object):
    def __init__(self, idx=None, prob=None):
        """Node class to build tree leaves.

        Keyword Arguments:
            idx {list} -- 1d list object with int (default: {None})
            left {Node Object} -- Left child
            right {Node Object} -- Right child
            feature {int} -- Feature number
            split {float} -- Split point of x
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
        """List length, positive probability and rate when x is splited into two pieces

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- 1d list object with int
            feature {int} -- Feature number
            split {float} -- Split point of x

        Returns:
            named tuple -- split effect
        """

        pos_cnt = [0, 0]
        cnt = [0, 0]
        idx_split = [[], []]
        # Iterate each row and compare with the split point
        for i in idx:
            xi, yi = X[i][feature], y[i]
            if xi < split:
                idx_split[0].append(i)
                cnt[0] += 1
                pos_cnt[0] += yi
            else:
                idx_split[1].append(i)
                cnt[1] += 1
                pos_cnt[1] += yi
        # Calculate the split effect
        n = len(idx)
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
            int or float -- The best choice of gain, split and split effect
        """
        unique = set([X[i][feature] for i in idx])
        # In case of empty split
        unique.remove(min(unique))
        info = self._get_info(y, idx)
        # Make sure any gain will be larger than initial value
        gain = - float('inf')
        split = None
        se = None
        for _split in unique:
            # Prepare the input needed to calculate condtional info
            _se = self._get_split_effect(X, y, idx, feature, _split)
            # Calculate then conditional info and gain
            cond_info = self._get_cond_info(_se)
            # Check if we got larger gain
            _gain = info - cond_info
            if _gain > gain:
                gain = _gain
                split = _split
                se = _se
        return gain, split, se

    def _choose_feature(self, X, y, idx):
        """Choose the feature which has max info gain

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- 1d list object with int

        Returns:
            tuple -- (feature number, split point, split effect)
        """
        # gains = []
        m = len(X[0])
        gain = - float('inf')
        feature = None
        split = None
        se = None
        # Compare the info gain of each feature and choose best one.
        for _feature in range(m):
            _gain, _split, _se = self._choose_split_point(X, y, idx, _feature)
            # gains.append(_gain)
            if _gain > gain:
                gain = _gain
                feature = _feature
                split = _split
                se = _se
        # print("gains", gains)
        return feature, split, se

    def get_rules(self):
        """[summary]

        Returns:
            [type] -- [description]
        """

        que = [[self.root, []]]
        # format of rule [[Feature, op, split], prob]
        self.rules = []
        while que:
            nd, rule = que.pop()
            if nd.left is None and nd.right is None:
                self.rules.append([rule, nd.prob])
            if nd.left is not None:
                rule_left = copy(rule)
                rule_left.append([nd.feature, 0, nd.split])
                que.append([nd.left, rule_left])
            if nd.right is not None:
                rule_right = copy(rule)
                rule_right.append([nd.feature, 1, nd.split])
                que.append([nd.right, rule_right])

    def fit(self, X, y, max_depth=3, min_samples_split=2):
        """Build a decision tree classifier.

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
            if depth == max_depth:
                break
            if len(nd.idx) < min_samples_split:
                continue
            # print("Input idx is:", len(nd.idx))
            nd.feature, nd.split, se = self._choose_feature(X, y, nd.idx)

            # print("Input Feature and split:", nd.feature, nd.split)
            # print("Output idx is:", len(se.idx[0]), len(se.idx[1]))
            # print("prob", se.prob)
            nd.left = Node(se.idx[0], se.prob[0])
            nd.right = Node(se.idx[1], se.prob[1])
            que.append([depth+1, nd.left])
            que.append([depth+1, nd.right])
        clf.height = depth
        clf.get_rules()

    def _print_rules(self, expr):
        """[summary]

        Arguments:
            expr {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        feathure, op, split = expr
        op = ">=" if op else "<"
        return "Feature%d %s %.4f" % (feathure, op, split)

    def print_rules(self):
        """[summary]
        """

        for i, rule in enumerate(self.rules):
            exprs, prob = rule
            print("Rule %d: " % i, ' | '.join(
                map(self._print_rules, exprs)) + ' => Prob %.4f' % prob)

    def Expr(self, expr, row):
        """[summary]

        Arguments:
            expr {[type]} -- [description]
            row {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        feathure, op, split = expr
        return (row[feathure] - split) * (1 if op else -1) >= 0

    def predict_prob(self, X):
        """[summary]

        Arguments:
            X {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        y = []
        for row in X:
            for rule in self.rules:
                exprs, prob = rule
                if all(self.Expr(expr, row) for expr in exprs):
                    y.append(prob)
                    break
        return y

    def predict(self, X, threshold=0.5):
        """[summary]

        Arguments:
            X {[type]} -- [description]

        Keyword Arguments:
            threshold {float} -- [description] (default: {0.5})

        Returns:
            [type] -- [description]
        """

        return [int(y >= threshold) for y in self.predict_prob(X)]


if __name__ == "__main__":
    # Load data
    from load_data import load_breast_cancer
    from sys import getsizeof
    X, y = load_breast_cancer()

    # Train model
    clf = DecisionTree()
    clf.fit(X, y)

    # Show rules
    clf.print_rules()

    # Model accuracy
    acc = sum((y_hat == y for y_hat, y in zip(clf.predict(X), y))) / len(y)
    print("Accuracy is: %.2f" % acc * 100)
    print(getsizeof(clf))
