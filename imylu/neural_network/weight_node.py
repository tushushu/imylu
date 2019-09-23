"""
@Author: tushushu
@Date: 2019-09-02 19:26:45
"""
from typing import Tuple, Union
import numpy as np
from .input_node import BaseNode


class WeightNode(BaseNode):
    """Weight node class.

    Attributes:
        name {str} -- The name of Node.
        value {Optional[float]} -- The value of Node.
        inbound_nodes {List[Node]]} -- inbound nodes.
        outbound_nodes {List[Node]} -- outbound nodes.
        gradients {Dict[Node, float]} -- Keys: inbound nodes, Values: gradients.
    """
    def __init__(self, shape: Union[Tuple[int, int], int], name=None, learning_rate=None):
        """Initialize a node instance and connect inbound nodes to this instance.

        Arguments:
            shape {Union[Tuple[int, int], int]} -- The shape of value.
            data {InputNode} -- The value of data is an ndarray with shape(m, n).
            weights {InputNode} -- The value of weights is an ndarray with shape(n, k).
            bias {InputNode} -- The value of bias is an ndarray with shape(1, k).
            learning_rate {float} -- The learning rate of nerual network.
        """
        BaseNode.__init__(self, name=name)
        if isinstance(shape, int):
            self.value = np.zeros(shape)
        if isinstance(shape, tuple):
            self.value = np.random.randn(*shape)
        self.learning_rate = learning_rate

    def forward(self):
        """Forward the input of inbound_nodes."""
        pass

    def backward(self):
        """Backward the gradient of outbound_nodes."""
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            self.gradients[self] += node.gradients[self]
        partial = self.gradients[self]
        self.value -= partial * self.learning_rate
