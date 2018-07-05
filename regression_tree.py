# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-07-05 17:51:04 
@Last Modified by: tushushu 
@Last Modified time: 2018-07-05 17:51:04 
"""
from copy import copy
from utils import load_boston_house_prices, train_test_split, get_r2, run_time, min_max_scale


class Node(object):
    def __init__(self, avg=None):
        """Node class to build tree leaves.

        Keyword Arguments:
            avg {float} -- prediction of y (default: {None})
        """
        self.avg = avg

        self.left = None
        self.right = None
        self.feature = None
        self.split = None


class RegressionTree(object):
    def __init__(self):
        """RegressionTree class.

        Attributes:
            root: the root node of DecisionTree
            height: the height of DecisionTree
        """

        self.root = Node()
        self.height = 0

    def _get_split_var(self, X, y, idx, feature, split):
        """Calculate the average of each set when x is splitted into two pieces.
        D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2 

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number
            split {float} -- Split point of x

        Returns:
            list -- Average of splitted x
        """

        y_sum = [0, 0]
        y_cnt = [0, 0]
        y_sqr_sum = [0, 0]
        # Iterate each row and compare with the split point
        for i in idx:
            xi, yi = X[i][feature], y[i]
            if xi < split:
                y_cnt[0] += 1
                y_sum[0] += yi
                y_sqr_sum[0] += yi ** 2
            else:
                y_cnt[1] += 1
                y_sum[1] += yi
                y_sqr_sum[1] += yi ** 2
        # Calculate the variance of y
        y_avg = [y_sum[0] / y_cnt[0], y_sum[1] / y_cnt[1]]
        y_var = [y_sqr_sum[0] - y_sum[0] * y_avg[0],
                 y_sqr_sum[1] - y_sum[1] * y_avg[1]]
        return sum(y_var), split, y_avg

    def _choose_split_point(self, X, y, idx, feature):
        """Iterate each xi and split x, y into two pieces,
        and the best split point is the xi when we get min variance.

        Arguments:
            x {list} -- 1d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number

        Returns:
            tuple -- The best choice of variance, feature, split point and average
        """
        # Feature cannot be splitted if there's only one unique element.
        unique = set([X[i][feature] for i in idx])
        if len(unique) == 1:
            return None
        # In case of empty split
        unique.remove(min(unique))
        # Get split point which has min var
        y_var, split, y_avg = min((self._get_split_var(X, y, idx, feature, split)
                                   for split in unique), key=lambda x: x[0])
        return y_var, feature, split, y_avg

    def _choose_feature(self, X, y, idx):
        """Choose the feature which has min variance.

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- indexes, 1d list object with int

        Returns:
            tuple -- (feature number, split point, average, idx_split)
        """

        m = len(X[0])
        # Compare the info gain of each feature and choose best one.
        split_rets = [x for x in map(lambda x: self._choose_split_point(
            X, y, idx, x), range(m)) if x is not None]
        # Terminate if no feature can be splitted
        if split_rets == []:
            return None
        _, feature, split, y_avg = min(
            split_rets, key=lambda x: x[0])
        # Get split idx into two pieces and empty idx
        idx_split = [[], []]
        while idx:
            i = idx.pop()
            xi = X[i][feature]
            if xi < split:
                idx_split[0].append(i)
            else:
                idx_split[1].append(i)
        return feature, split, y_avg, idx_split

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
            Rule: 2D list like [[Feature, op, split], avg]
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
                self.rules.append([literals, nd.avg])
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
            y {list} -- 1d list object with float

        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {2})
            min_samples_split {int} -- The minimum number of samples required to split an internal node (default: {2})
        """

        # Initialize with depth, node, indexes
        self.root = Node()
        que = [[0, self.root, list(range(len(y)))]]
        # Breadth-First Search
        while que:
            depth, nd, idx = que.pop(0)
            # Terminate loop if tree depth is more than max_depth
            if depth == max_depth:
                break
            # Stop split when number of node samples is less than min_samples_split or Node is 100% pure.
            if len(idx) < min_samples_split or set(map(lambda i: y[i], idx)) == 1:
                continue
            # Stop split if no feature has more than 2 unique elements
            feature_rets = self._choose_feature(X, y, idx)
            if feature_rets is None:
                continue
            # Split
            nd.feature, nd.split, y_avg, idx_split = feature_rets
            nd.left = Node(y_avg[0])
            nd.right = Node(y_avg[1])
            que.append([depth+1, nd.left, idx_split[0]])
            que.append([depth+1, nd.right, idx_split[1]])
        # Update tree depth and rules
        self.height = depth
        self._get_rules()

    def print_rules(self):
        """Print the rules of all the regression decision tree leaf nodes.
        """

        for i, rule in enumerate(self.rules):
            literals, avg = rule
            print("Rule %d: " % i, ' | '.join(
                literals) + ' => y_hat %.4f' % avg)

    def _predict(self, row):
        """Auxiliary function of predict.

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
        return nd.avg

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with float
        """

        return [self._predict(Xi) for Xi in X]


@run_time
def main():
    print("Tesing the accuracy of LinearRegression(batch)...")
    # Load data
    X, y = load_boston_house_prices()
    X = min_max_scale(X)
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    reg = RegressionTree()
    reg.fit(X=X_train, y=y_train, max_depth=3)

    # _X = [[x] for x in range(10)]
    # _y = [5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05]
    # reg.fit(_X, _y, 1)

    # Model accuracy
    get_r2(reg, X_test, y_test)
    reg.print_rules()


if __name__ == "__main__":
    main()
