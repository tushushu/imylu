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
        data {Optional(InputNode)} -- Features data.
        prediction {Optional(LinearNode)} -- Predction data.
    """

    def __init__(self):
        self.nodes_sorted = []
        self._learning_rate = None
        self.data = None
        self.prediction = None
        self.label = None

    def __str__(self):
        if not self.nodes_sorted:
            return "Network has not be trained yet!"
        print("Network informantion:\n")
        ret = ["learning rate:", str(self._learning_rate), "\n"]
        for node in self.nodes_sorted:
            ret.append(node.name)
            ret.append(str(node.value.shape))
            ret.append("\n")
        return " ".join(ret)

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
            input_nodes {List[InputNode]} -- All the input nodes without inbound_nodes.
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

    def build_network(self, data: ndarray, label: ndarray, n_hidden: int, n_feature: int):
        """[summary]

        Arguments:
            data {ndarray} -- [description]
            label {ndarray} -- [description]
            n_hidden {int} -- [description]
            n_feature {int} -- [description]
        """
        weight_node1 = WeightNode(shape=(n_feature, n_hidden), name="W1")
        bias_node1 = WeightNode(shape=n_hidden, name="b1")
        weight_node2 = WeightNode(shape=(n_hidden, 1), name="W2")
        bias_node2 = WeightNode(shape=1, name="b2")
        self.data = InputNode(data, name="X")
        self.label = InputNode(label, name="y")
        linear_node1 = LinearNode(
            self.data, weight_node1, bias_node1, name="l1")
        sigmoid_node1 = SigmoidNode(linear_node1, name="s1")
        self.prediction = LinearNode(
            sigmoid_node1, weight_node2, bias_node2, name="prediction")
        MseNode(self.label, self.prediction, name="mse")
        input_nodes = [weight_node1, bias_node1,
                       weight_node2, bias_node2, self.data, self.label]
        self.topological_sort(input_nodes)

    def train_network(self, label: ndarray, epochs: int, n_sample: int, batch_size: int):
        """[summary]

        Arguments:
            label {ndarray} -- [description]
            epochs {int} -- [description]
            n_sample {int} -- [description]
            batch_size {int} -- [description]
        """
        steps_per_epoch = n_sample // batch_size
        for i in range(epochs):
            loss = 0
            for _ in range(steps_per_epoch):
                indexes = choice(n_sample, batch_size, replace=True)
                self.data.indexes = indexes
                self.label.indexes = indexes
                self.forward_and_backward()
                loss += self.nodes_sorted[-1].value
            print("Epoch: {}, Loss: {:.3f}".format(
                i + 1, loss / steps_per_epoch))
        print()

    def pop_unused_nodes(self):
        """[summary]
        """
        for _ in range(len(self.nodes_sorted)):
            node = self.nodes_sorted.pop(0)
            if node.name in ("mse", "y"):
                continue
            self.nodes_sorted.append(node)

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
        # Construct nerual network.
        self.build_network(data, label, n_hidden, n_feature)
        self.learning_rate = learning_rate
        print("Total number of samples = {}".format(n_sample))
        # Train network.
        self.train_network(label, epochs, n_sample, batch_size)
        # Pop unused node for predition.
        self.pop_unused_nodes()

    def predict(self, data: ndarray) -> ndarray:
        """Get the prediction of label.

        Arguments:
            data {ndarray} -- Testing data.

        Returns:
            ndarray -- Prediction of label.
        """

        self.data.value = data
        self.data.indexes = range(data.shape[0])
        self.forward()
        return self.prediction.value.flatten()
