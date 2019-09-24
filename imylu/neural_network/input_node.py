"""
@Author: tushushu
@Date: 2019-05-30 14:38:56
"""
from numpy import ndarray
from numpy.random import choice
from .base_node import BaseNode


class InputNode(BaseNode):
    """Input node class.

    Attributes:
        name {str} -- The name of Node.
        value {Optional[float]} -- The value of Node.
        inbound_nodes {List[Node]]} -- inbound nodes.
        outbound_nodes {List[Node]} -- outbound nodes.
        gradients {Dict[Node, float]} -- Keys: inbound nodes, Values: gradients.
    """
    def __init__(self, value: ndarray, name=None):
        """Initialize a node instance and connect inbound nodes to this instance.

        Arguments:
            value {ndarray} -- An ndarray with shape(m, n).

        Keyword Arguments:
            name {str} -- The name of Node. (default: {None})
        """
        BaseNode.__init__(self, name=name)
        self.value = value
        self.indexes = None

    @property
    def value(self):
        err_msg = "Indexes is None!"
        assert self.indexes is not None, err_msg
        return self._value[self.indexes]

    @value.setter
    def value(self, value: ndarray):
        BaseNode.value.fset(self, value)

    def forward(self):
        """Forward the input of inbound_nodes."""
        return

    def backward(self):
        """Backward the gradient of outbound_nodes."""
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            self.gradients[self] += node.gradients[self]
