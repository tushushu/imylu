# -*- coding: utf-8 -*-
"""
@Author: tushushu 
@Date: 2018-06-13 14:55:11 
@Last Modified by: tushushu 
@Last Modified time: 2018-06-13 14:55:11 
"""

from math import log2
from copy import copy
from utils import load_breast_cancer, train_test_split, get_acc, run_time


class Node(object):
    def __init__(self, prob=None):
        """Node class to build tree leaves.

        Keyword Arguments:
            prob {float} -- positive probability (default: {None})
        """
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
        """

        self.root = Node()
        self.height = 0

    def _get_split_effect(self, X, y, idx, feature, split):
        """List length, positive probability and rate when x is splitted into two pieces.

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number
            split {float} -- Split point of x

        Returns:
            tuple -- prob, rate
        """

        n = len(idx)
        pos_cnt = [0, 0]
        cnt = [0, 0]

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

        return prob, rate

    def _get_entropy(self, p):
        """Calculate entropy
        Probability:
        P(X=x_i) = p_i, i <- [1, n]

        Entropy: 
        H(p) = -Sum(p_i * log(p_i)), i <- [1, n]

        Take binary classifaction for exmaple, 
        Likelihood function, yi <- y, and p is a constant:
        Likelihood = Product(p^yi * (1-p)^(1-yi))

        Take the logarithm of both sides of this equation:
        L = Sum(yi * logp + (1-yi) * log(1-p))

        L / m = Sum(yi/m * logp + (1-yi) / m * log(1-p))
        L / m = Sum(p * logp + (1-p) * log(1-p))
        L / m = H(p)

        So Maximising the Entropy equals to Maximising the likelihood
        -------------------------------------------------------------

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
        Probability:
        P(y=y_i) = p_i, i <- [1, n]

        Entropy: 
        Info(y) = H(p) = -Sum(p_i * log(p_i)), i <- [1, n]
        --------------------------------------------------

        Arguments:
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- indexes, 1d list object with int

        Returns:
            float -- Info of y
        """

        p = sum(y[i] for i in idx) / len(idx)
        return self._get_entropy(p)

    def _get_cond_info(self, prob, rate):
        """Calculate conditonal info of x, y
        Conditional Probability:
        Suppose there are k cases:
        P(A = A_i), i <- [1, k]

        Entropy: 
        CondInfo(X, y) = -Sum(p_i * H(y | A = A_i)), i <- [1, k]
        -------------------------------------------------------

        Arguments:
            prob {list} -- [left node probability, right node probability]
            rate {list} -- [left node positive rate, right node positive rate]

        Returns:
            float -- Conditonal info of x
        """

        info_left = self._get_entropy(prob[0])
        info_right = self._get_entropy(prob[1])
        return rate[0] * info_left + rate[1] * info_right

    def _choose_split(self, X, y, idx, feature):
        """Iterate each xi and split x, y into two pieces,
        and the best split point is the xi when we get max gain.
        Info gain:
        Gain(X, y) = Info(y) - CondInfo(X, y)
        ---------------------------------------------------------

        Arguments:
            x {list} -- 1d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number

        Returns:
            tuple -- The best choice of gain, feature, split point and probability
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
            prob, rate = self._get_split_effect(
                X, y, idx, feature, split)
            cond_info = self._get_cond_info(prob, rate)
            gain = info - cond_info
            return gain, split, prob
        # Get split point which has max gain
        gain, split, prob = max((f(split)
                                 for split in unique), key=lambda x: x[0])
        return gain, feature, split, prob

    def _choose_feature(self, X, y, idx):
        """Choose the feature which has max info gain.

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int 0 or 1
            idx {list} -- indexes, 1d list object with int

        Returns:
            tuple -- (feature number, split point, probability)
        """

        m = len(X[0])
        # Compare the info gain of each feature and choose best one.
        split_rets = map(lambda x: self._choose_split(X, y, idx, x), range(m))
        split_rets = filter(lambda x: x is not None, split_rets)
        # Return None if no feature can be splitted
        return max(split_rets, default=None, key=lambda x: x[0])

    def _split_feature(self, X, idxs, feature, split):
        """Split indexes into two arrays according to split point.

        Arguments:
            X {list} -- 2d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number
            split {float} -- Split point of the feature

        Returns:
            list -- [left idx, right idx]
        """

        idxs_split = [[], []]
        for idx in idxs:
            xi = X[idx][feature]
            if xi < split:
                idxs_split[0].append(idx)
            else:
                idxs_split[1].append(idx)
        return idxs_split

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

        # Initialize with depth, node, indexes
        self.root = Node()
        que = [[0, self.root, list(range(len(y)))]]
        # Breadth-First Search
        while que:
            depth, nd, idxs = que.pop(0)
            # Terminate loop if tree depth is more than max_depth
            if depth == max_depth:
                break
            # Stop split when number of node samples is less than min_samples_split or Node is 100% pure.
            if len(idxs) < min_samples_split or nd.prob == 1 or nd.prob == 0:
                continue
            # Stop split if no feature has more than 2 unique elements
            split_ret = self._choose_feature(X, y, idxs)
            if split_ret is None:
                continue
            # Split
            _, nd.feature, nd.split, prob = split_ret
            idxs_split = self._split_feature(X, idxs, nd.feature, nd.split)
            nd.left = Node(prob[0])
            nd.right = Node(prob[1])
            que.append([depth+1, nd.left, idxs_split[0]])
            que.append([depth+1, nd.right, idxs_split[1]])
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


@run_time
def main():
    print("Tesing the accuracy of DecisionTree...")
    # Load data
    X, y = load_breast_cancer()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    clf = DecisionTree()
    clf.fit(X_train, y_train, max_depth=4)
    # Show rules
    clf.print_rules()
    # Model accuracy
    get_acc(clf, X_test, y_test)


if __name__ == "__main__":
    main()
