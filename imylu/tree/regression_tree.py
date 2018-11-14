# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-05 17:51:04
@Last Modified by: tushushu
@Last Modified time: 2018-07-05 17:51:04
"""
from copy import copy
from ..utils.utils import list_split


class Node(object):
    def __init__(self, score=None):
        """Node class to build tree leaves.

        Attributes:
            score {float} -- prediction of y (default: {None})
            left {Node} -- Left child node
            right {Node} -- Right child node
            feature {int} -- Column index
            split {int} --  Split point
        """
        self.score = score

        self.left = None
        self.right = None
        self.feature = None
        self.split = None


class RegressionTree(object):
    def __init__(self):
        """RegressionTree class.

        Attributes:
            root: the root node of DecisionTree
            depth: the depth of DecisionTree
        """

        self.root = Node()
        self.depth = 1
        self._rules = None

    def _get_split_mse(self, X, y, idx, feature, split):
        """Calculate the mse of each set when x is splitted into two pieces.
        MSE as Loss fuction:
        y_hat = Sum(y_i) / n, i <- [1, n]
        Loss(y_hat, y) = Sum((y_hat - y_i) ^ 2), i <- [1, n]
        Loss = LossLeftNode+ LossRightNode
        --------------------------------------------------------------------

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number
            split {float} -- Split point of x

        Returns:
            tuple -- MSE, split point and average of splitted x
        """

        split_sum = [0, 0]
        split_cnt = [0, 0]
        split_sqr_sum = [0, 0]
        # Iterate each Xi and compare with the split point
        for i in idx:
            xi, yi = X[i][feature], y[i]
            if xi < split:
                split_cnt[0] += 1
                split_sum[0] += yi
                split_sqr_sum[0] += yi ** 2
            else:
                split_cnt[1] += 1
                split_sum[1] += yi
                split_sqr_sum[1] += yi ** 2
        # Calculate the mse of y, D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2
        split_avg = [split_sum[0] / split_cnt[0], split_sum[1] / split_cnt[1]]
        split_mse = [split_sqr_sum[0] - split_sum[0] * split_avg[0],
                     split_sqr_sum[1] - split_sum[1] * split_avg[1]]
        return sum(split_mse), split, split_avg

    def _choose_split(self, X, y, idx, feature):
        """Iterate each xi and split x, y into two pieces,
        and the best split point is the xi when we get minimum mse.

        Arguments:
            x {list} -- 1d list object with int or float
            y {list} -- 1d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number

        Returns:
            tuple -- The best choice of mse, feature, split point and average
        """
        # Feature cannot be splitted if there's only one unique element.
        unique = set([X[i][feature] for i in idx])
        if len(unique) == 1:
            return None
        # In case of empty split
        unique.remove(min(unique))
        # Get split point which has min mse
        mse, split, split_avg = min(
            (self._get_split_mse(X, y, idx, feature, split)
             for split in unique), key=lambda x: x[0])
        return mse, feature, split, split_avg

    def _choose_feature(self, X, y, idx):
        """Choose the feature which has minimum mse.

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int or float
            idx {list} -- indexes, 1d list object with int

        Returns:
            tuple -- (feature number, split point, average, idx_split)
        """

        m = len(X[0])
        # Compare the mse of each feature and choose best one.
        split_rets = map(lambda j: self._choose_split(X, y, idx, j), range(m))
        split_rets = filter(lambda x: x is not None, split_rets)
        # Terminate if no feature can be splitted
        return min(split_rets, default=None, key=lambda x: x[0])

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
            Rule: 2D list like [[Feature, op, split], score]
            Op: -1 means less than, 1 means equal or more than
        """

        que = [[self.root, []]]
        self._rules = []
        # Breadth-First Search
        while que:
            nd, exprs = que.pop(0)
            # Generate a rule when the current node is leaf node
            if not(nd.left or nd.right):
                # Convert expression to text
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, nd.score])
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

    def fit(self, X, y, max_depth=5, min_samples_split=2):
        """Build a regression decision tree.
        Note:
            At least there's one column in X has more than 2 unique elements
            y cannot be all the same value

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int or float

        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {5})
            min_samples_split {int} -- The minimum number of samples required
            to split an internal node (default: {2})
        """

        # Initialize with depth, node, indexes
        self.root.score = sum(y) / len(y)
        idxs = list(range(len(y)))
        que = [(self.depth + 1, self.root, idxs)]
        # Breadth-First Search
        while que:
            depth, nd, idxs = que.pop(0)
            # Terminate loop if tree depth is more than max_depth
            if depth > max_depth:
                depth -= 1
                break
            # Stop split when number of node samples is less than
            # min_samples_split or Node is 100% pure.
            if len(idxs) < min_samples_split or \
                    all(map(lambda i: y[idxs[0]] == y[i], idxs)):
                continue
            # Stop split if no feature has more than 2 unique elements
            split_ret = self._choose_feature(X, y, idxs)
            if split_ret is None:
                continue
            # Split
            _, feature, split, split_avg = split_ret
            # Update properties of current node
            nd.feature = feature
            nd.split = split
            nd.left = Node(split_avg[0])
            nd.right = Node(split_avg[1])
            # Put children of current node in que
            idxs_split = list_split(X, idxs, feature, split)
            que.append((depth + 1, nd.left, idxs_split[0]))
            que.append((depth + 1, nd.right, idxs_split[1]))
        # Update tree depth and rules
        self.depth = depth
        self._get_rules()

    @property
    def rules(self):
        """Print the rules of all the regression decision tree leaf nodes.
        """

        for i, rule in enumerate(self._rules):
            literals, score = rule
            print("Rule %d: " % i, ' | '.join(
                literals) + ' => y_hat %.4f' % score)

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1D list with int or float

        Returns:
            int or float -- prediction of yi
        """

        nd = self.root
        while nd.left and nd.right:
            if Xi[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd.score

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with int or float
        """

        return [self._predict(Xi) for Xi in X]
