"""
@Author: tushushu
@Date: 2019-05-29 15:23:35
"""
from abc import abstractmethod, ABC
from typing import List, Dict
from numpy import ndarray


class BaseNode(ABC):
    """BaseNode class.

    Attributes:
        name {str} -- The name of Node.
        value {Optional[float]} -- The value of Node.
        inbound_nodes {List[Node]]} -- inbound nodes.
        outbound_nodes {List[Node]} -- outbound nodes.
        gradients {Dict[Node, float]} -- Keys: inbound nodes, Values: gradients.
    """

    def __init__(self, *inbound_nodes, name=None):
        """Initialize a node instance and connect inbound nodes to this instance.

        Keyword Arguments:
            name {str} -- The name of Node. (default: {None})
        """
        self.name = name
        self._value = None
        self.inbound_nodes = [x for x in inbound_nodes]  # type: List[BaseNode]
        self.outbound_nodes = []  # type: List[BaseNode]
        self.gradients = dict()  # type: Dict[BaseNode, float]
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

    def __str__(self):
        size = str(self.value.shape) if self.value is not None else "null"
        return "<Node name: %s, Node size: %s>" % (self.name, size)

    @property
    def value(self)->ndarray:
        """Protect the attributes of value.

        Returns:
            ndarray
        """
        return self._value

    @value.setter
    def value(self, value):
        err_msg = "'value' has to be a number or a numpy array!"
        assert isinstance(value, (ndarray, int, float)), err_msg
        self._value = value

    @abstractmethod
    def forward(self):
        """Forward the input of inbound_nodes."""
        return

    @abstractmethod
    def backward(self):
        """Backward the gradient of outbound_nodes."""
        return
