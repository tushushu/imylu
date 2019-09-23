"""
@Author: tushushu
@Date: 2019-09-02 14:24:30
"""
from copy import copy
from numpy import ndarray
from numpy.random import choice
from .input_node import InputNode
from .linear_node import LinearNode
from .sigmoid_node import SigmoidNode
from .mse_node import MseNode
from .weight_node import WeightNode


class DNN:
    """DNN class.

    Attributes:
        nodes_sorted {list} -- All the nodes of nerual-network sorted for training.
        learning_rate {float} -- Learning rate.
    """

    def __init__(self):
        self.nodes_sorted = []
        self._learning_rate = None
        self.data = None
        self.prediction = None

    @property
    def learning_rate(self) -> float:
        """Protect the learning_rate property.

        Returns:
            float
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        for node in self.nodes_sorted:
            if isinstance(node, WeightNode):
                node.learning_rate = learning_rate

    def topological_sort(self, input_nodes):
        """Sort the nodes of nerual-network so as to forward and backward.

        Arguments:
            input_nodes {List[InputNode]}
        """
        nodes_sorted = []
        que = copy(input_nodes)
        unique = set()
        while que:
            node = que.pop(0)
            nodes_sorted.append(node)
            unique.add(node)
            for outbound_node in node.outbound_nodes:
                if all(x in unique for x in outbound_node.inbound_nodes):
                    que.append(outbound_node)
        self.nodes_sorted = nodes_sorted

    def forward(self):
        """Forward the values of all the nodes.
        """
        assert self.nodes_sorted is not None, "nodes_sorted is empty!"
        for node in self.nodes_sorted:
            node.forward()

    def backward(self):
        """Backward the gradients of all the nodes.
        """
        assert self.nodes_sorted is not None, "nodes_sorted is empty!"
        for node in self.nodes_sorted[::-1]:
            node.backward()

    def forward_and_backward(self):
        """Forward the values and backward the gradients of all the nodes.
        """
        self.forward()
        self.backward()

    def fit(self, data: ndarray, label: ndarray, n_hidden: int, epochs: int,
            batch_size: int, learning_rate: float):
        """Train DNN model.

        Arguments:
            data {ndarray} -- Features, 2d ndarray.
            label {ndarray} -- Label, 1d ndarray.
            n_hidden {int} -- Number of hidden nodes.
            epochs {int} -- Number of training iterations.
            batch_size {int} -- Batch size for mini-batch training.
            learning_rate {float} -- Learning rate for training.
        """

        label = label.reshape(-1, 1)
        n_sample, n_feature = data.shape
        steps_per_epoch = n_sample // batch_size
        W1 = WeightNode(shape=(n_feature, n_hidden), name="W1")
        b1 = WeightNode(shape=n_hidden, name="b1")
        W2 = WeightNode(shape=(n_hidden, 1), name="W2")
        b2 = WeightNode(shape=1, name="b2")
        self.data = InputNode(data, name="X")
        y = InputNode(label, name="y")
        l1 = LinearNode(self.data, W1, b1, name="l1")
        s1 = SigmoidNode(l1, name="s1")
        self.prediction = LinearNode(s1, W2, b2, name="prediction")
        mse = MseNode(y, self.prediction, name="mse")
        input_nodes = [W1, b1, W2, b2, self.data, y]
        self.topological_sort(input_nodes)
        self.learning_rate = learning_rate
        print("Total number of samples = {}".format(n_sample))

        for i in range(epochs):
            loss = 0
            for _ in range(steps_per_epoch):
                indexes = choice(n_sample, batch_size, replace=True)
                self.data.indexes = indexes
                y.indexes = indexes
                self.forward_and_backward()
                loss += self.nodes_sorted[-1].value
            print("Epoch: {}, Loss: {:.3f}".format(
                i + 1, loss / steps_per_epoch))

        for _ in range(len(self.nodes_sorted)):
            node = self.nodes_sorted.pop(0)
            if node.name in ("mse", "prediction"):
                continue
            self.nodes_sorted.append(node)

    def predict(self, data: ndarray) -> ndarray:
        """[summary]

        Arguments:
            data {ndarray} -- [description]

        Returns:
            ndarray -- [description]
        """

        self.data = InputNode(data, name="X")
        self.forward()
        return self.prediction
