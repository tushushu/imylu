"""
@Author: tushushu
@Date: 2019-05-30 14:46:27
"""
import numpy as np
from .base_node import BaseNode
from .linear_node import LinearNode
from .input_node import InputNode


class MseNode(BaseNode):
    """MSE node class.

    Attributes:
        name {str} -- The name of Node.
        value {Optional[float]} -- The value of Node.
        inbound_nodes {List[Node]]} -- inbound nodes.
        outbound_nodes {List[Node]} -- outbound nodes.
        gradients {Dict[Node, float]} -- Keys: inbound nodes, Values: gradients.
        n_label {int} -- Number of labels.
        diff {int} -- The difference between label and prediction. The value of
        diff is an ndarray with shape(n_label, 1).
    """
    def __init__(self, label: InputNode, pred: LinearNode, name=None):
        """Initialize a node instance and connect inbound nodes to this instance.

        Arguments:
            label {InputNode} -- The value of label is an ndarray with shape(n_label, ).
            pred {BaseNode} -- The value of pred is an ndarray with shape(n_label, ).
        """
        BaseNode.__init__(self, label, pred, name=name)
        self.n_label = None
        self.diff = None

    def forward(self):
        """Forward the input of inbound_nodes.

        MSE = (label - prediction) ^ 2 / n_label
        """
        label, pred = self.inbound_nodes
        self.n_label = label.value.shape[0]
        self.diff = (label.value - pred.value).reshape(-1, 1)
        self.value = np.mean(self.diff**2)

    def backward(self):
        """Backward the gradient of outbound_nodes.

        dMSE / dLabel = 2 * (label - prediction) / n_label
        dMSE / dPrediction = -2 * (label - prediction) / n_label
        """
        label, pred = self.inbound_nodes
        self.gradients[label] = (2 / self.n_label) * self.diff
        self.gradients[pred] = -self.gradients[label]
