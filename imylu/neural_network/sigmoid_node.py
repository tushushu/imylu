"""
@Author: tushushu
@Date: 2019-05-30 14:44:33
"""
import numpy as np
from numpy import ndarray
from .base_node import BaseNode
from .linear_node import LinearNode


class SigmoidNode(BaseNode):
    """Sigmoid node class.

    Attributes:
        name {str} -- The name of Node.
        value {Optional[float]} -- The value of Node.
        inbound_nodes {List[Node]]} -- inbound nodes.
        outbound_nodes {List[Node]} -- outbound nodes.
        gradients {Dict[Node, float]} -- Keys: inbound nodes, Values: gradients.
    """

    def __init__(self, input_node: LinearNode, name=None):
        """Initialize a node instance and connect inbound nodes to this instance.

        Arguments:
            input_node {LinearNode} -- The value of input_node is an ndarray with shape(m, n).
        """
        BaseNode.__init__(self, input_node, name=name)

    @staticmethod
    def _sigmoid(arr: ndarray) -> ndarray:
        """F(x) = 1 / (1 + e^(-x))

        Arguments:
            arr {ndarray} -- An ndarray with shape(m, n).

        Returns:
            ndarray -- An ndarray with shape(m, n).
        """
        return 1. / (1. + np.exp(-arr))

    @staticmethod
    def _derivative(arr: ndarray) -> ndarray:
        """F'(X) = F(X) * (1 - F(X))

        Arguments:
            arr {ndarray} -- An ndarray with shape(m, n).

        Returns:
            ndarray -- An ndarray with shape(m, n).
        """
        return arr * (1 - arr)

    def forward(self):
        """Forward the input of inbound_nodes.

        F(x) = 1 / (1 + e^(-x))
        """
        input_node = self.inbound_nodes[0]
        self.value = self._sigmoid(input_node.value)

    def backward(self):
        """Backward the gradient of outbound_nodes.

        dSigmoid / dX = Sigmoid * (1 - Sigmoid)
        """
        input_node = self.inbound_nodes[0]
        self.gradients = {input_node: np.zeros_like(input_node.value)}
        for output_node in self.outbound_nodes:
            grad_cost = output_node.gradients[self]
            self.gradients[input_node] += self._derivative(self.value) * grad_cost
