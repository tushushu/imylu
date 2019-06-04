# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-06-13 14:55:11
@Last Modified by: tushushu
@Last Modified time: 2019-05-28 10:26:11
"""
from copy import copy
from numpy import ndarray, log, apply_along_axis


class Node:
    """Node class to build tree leaves.

    Attributes:
        prob {float} -- Prediction of label. (default: {None})
        left {Node} -- Left child node.
        right {Node} -- Right child node.
        feature {int} -- Column index.
        split {int} --  Split point.
        gain {float} -- Information gain.
    """

    attr_names = ("prob", "left", "right", "feature", "split", "gain")

    def __init__(self, prob=None, left=None, right=None, feature=None, split=None, gain=None):
        self.prob = prob
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.gain = gain

    def __str__(self):
        ret = []
        for attr_name in self.attr_names:
            attr = getattr(self, attr_name)
            # Describe the attribute of Node.
            if attr is None:
                continue
            if isinstance(attr, Node):
                des = "%s: Node object." % attr_name
            else:
                des = "%s: %s" % (attr_name, attr)
            ret.append(des)

        return "\n".join(ret) + "\n"

    def copy(self, node):
        """Copy the attributes of another Node.

        Arguments:
            node {Node}
        """

        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)


class DecisionTree:
    """DecisionTree class only support binary classification with ID3.

    Attributes:
        root {Node} -- Root node of DecisionTree.
        depth {int} -- Depth of DecisionTree.
        _rules {list} -- Rules of all the tree nodes.
    """

    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None

    def __str__(self):
        ret = []
        for i, rule in enumerate(self._rules):
            literals, prob = rule

            ret.append("Rule %d: " % i + ' | '.join(
                literals) + ' => y_hat %.4f' % prob)
        return "\n".join(ret)

    @staticmethod
    def _expr2literal(expr: list) -> str:
        """Auxiliary function of get_rules.

        Arguments:
            expr {list} -- 1D list like [Feature, op, split].

        Returns:
            str
        """

        feature, operation, split = expr
        operation = ">=" if operation == 1 else "<"
        return "Feature%d %s %.4f" % (feature, operation, split)

    def get_rules(self):
        """Get the rules of all the tree nodes.
            Expr: 1D list like [Feature, op, split].
            Rule: 2D list like [[Feature, op, split], label].
            Op: -1 means less than, 1 means equal or more than.
        """

        # Breadth-First Search.
        que = [[self.root, []]]
        self._rules = []

        while que:
            node, exprs = que.pop(0)

            # Generate a rule when the current node is leaf node.
            if not(node.left or node.right):
                # Convert expression to text.
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, node.prob])

            # Expand when the current node has left child.
            if node.left:
                rule_left = copy(exprs)
                rule_left.append([node.feature, -1, node.split])
                que.append([node.left, rule_left])

            # Expand when the current node has right child.
            if node.right:
                rule_right = copy(exprs)
                rule_right.append([node.feature, 1, node.split])
                que.append([node.right, rule_right])

    @staticmethod
    def _get_info(prob_pos: float) -> float:
        """Calculate entropy of label.
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

        So Maximizing the Entropy equals to Maximizing the likelihood.
        -------------------------------------------------------------

        Arguments:
            prob_pos {float} -- Positive prob of label.

        Returns:
            float -- Entropy
        """

        # According to L'Hospital's rule, 0 * log2(0) equals to zero.
        if prob_pos in (0, 1):
            entropy = 0
        else:
            prob_neg = 1 - prob_pos
            entropy = -prob_pos * log(prob_pos) - prob_neg * log(prob_neg)

        return entropy

    def _get_cond_info(self, ratio_left: float, prob_left: float, prob_right: float) -> float:
        """Calculate conditonal info of x, y
        Suppose there are k cases, conditional Probability:
        P(A = A_i), i <- [1, k]

        Entropy:
        CondInfo(X, y) = -Sum(p_i * H(y | A = A_i)), i <- [1, k]

        Arguments:
            ratio_left {float} -- Conditional probability of left split.
            prob_left {float} -- Probability of left split.
            prob_right {float} -- Probability of right split.

        Returns:
            float -- Conditonal information.
        """

        info_left = self._get_info(prob_left)
        info_right = self._get_info(prob_right)
        return ratio_left * info_left + (1 - ratio_left) * info_right

    def _get_split_gain(self, col: ndarray, label: ndarray, split: float, info: float) -> Node:
        """Calculate the information gain of label when col is splitted into two pieces.
        Info gain:
        Gain(X, y) = Info(y) - CondInfo(X, y)
        --------------------------------------------------------------------

        Arguments:
            col {ndarray} -- A feature of training data.
            label {ndarray} -- Target values.
            split {float} -- Split point of column.
            info {float} -- Entropy of label.

        Returns:
            Node -- Information gain of label and prob of splitted x.
        """

        # Split label.
        left = col < split

        # Calculate ratio.
        ratio_left = left.sum() / len(col)

        # Calculate conditional information.
        prob_left = label[left].mean()
        prob_right = label[col >= split].mean()
        info_cond = self._get_cond_info(ratio_left, prob_left, prob_right)

        # Create nodes to store result.
        node = Node(split=split)
        node.gain = info - info_cond
        node.left = Node(prob_left)
        node.right = Node(prob_right)

        return node

    def _choose_split(self, col: ndarray, label: ndarray) -> Node:
        """Iterate each xi and split x, y into two pieces, and the best
        split point is the xi when we get maximum information gain.

        Arguments:
            col {ndarray} -- A feature of training data.
            label {ndarray} -- Target values.

        Returns:
            Node -- The best choice of information gain, split point and prob.
        """

        # Feature cannot be splitted if there's only one unique element.
        node = Node()
        unique = set(col)
        if len(unique) == 1:
            return node

        # In case of empty split.
        unique.remove(min(unique))

        # Calculate info
        info = self._get_info(label.mean())

        # Get split point which has max info gain.
        ite = map(lambda x: self._get_split_gain(col, label, x, info), unique)
        node = max(ite, key=lambda x: x.gain)

        return node

    def _choose_feature(self, data: ndarray, label: ndarray) -> Node:
        """Choose the feature which has maximum info gain.

        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.

        Returns:
            Node -- feature number, split point, prob.
        """

        # Compare the mse of each feature and choose best one.
        _ite = map(lambda x: (self._choose_split(data[:, x], label), x),
                   range(data.shape[1]))
        ite = filter(lambda x: x[0].split is not None, _ite)

        # Return None if no feature can be splitted.
        node, feature = max(
            ite, key=lambda x: x[0].gain, default=(Node(), None))
        node.feature = feature

        return node

    def fit(self, data: ndarray, label: ndarray, max_depth=4, min_samples_split=2):
        """Build a decision tree classifier.
        Note:
            At least there's one column in data has more than 2 unique elements,
            and label cannot be all the same value.

        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.

        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {4})
            min_samples_split {int} -- The minimum number of samples required
            to split an internal node. (default: {2})
        """

        # Initialize with depth, node, indexes.
        self.root.prob = label.mean()
        que = [(self.depth + 1, self.root, data, label)]

        # Breadth-First Search.
        while que:
            depth, node, _data, _label = que.pop(0)

            # Terminate loop if tree depth is more than max_depth.
            if depth > max_depth:
                depth -= 1
                break

            # Stop split when number of node samples is less than
            # min_samples_split or Node is 100% pure.
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue

            # Stop split if no feature has more than 2 unique elements.
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue

            # Copy the attributes of _node to node.
            node.copy(_node)

            # Put children of current node in que.
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            que.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))

        # Update tree depth and rules.
        self.depth = depth
        self.get_rules()

    def predict_one_prob(self, row: ndarray) -> float:
        """Auxiliary function of predict_prob.

        Arguments:
            row {ndarray} -- A sample of testing data.

        Returns:
            float -- Prediction of label.
        """

        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right

        return node.prob

    def predict_prob(self, data: ndarray) -> ndarray:
        """Get the probability of label.

        Arguments:
            data {ndarray} -- Testing data.

        Returns:
            ndarray -- Probabilities of label.
        """

        return apply_along_axis(self.predict_one_prob, axis=1, arr=data)

    def predict(self, data: ndarray, threshold=0.5):
        """Get the prediction of label.

        Arguments:
            data {ndarray} -- Testing data.

        Keyword Arguments:
            threshold {float} -- (default: {0.5})

        Returns:
            ndarray -- Prediction of label.
        """

        prob = self.predict_prob(data)
        return (prob >= threshold).astype(int)
