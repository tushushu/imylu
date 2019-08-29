"""
@Author: tushushu
@Date: 2019-05-29 15:23:35
"""

from typing import List, Dict


class BaseNode:
    """BaseNode class.

    Arguments:
        value {Optional[float]} -- The value of Node.
        inbound_nodes {List[Node]]} -- inbound nodes.
        outbound_nodes {List[Node]} -- outbound nodes.
        gradients {Dict[Node, float]} -- Keys: inbound nodes, Values: gradients.
    """
    def __init__(self, *inbound_nodes):
        """Initialize a node instance and connect inbound nodes to this instance."""
        self.value = None
        self.inbound_nodes = list(inbound_nodes) if inbound_nodes else []  # type: List[BaseNode]
        self.outbound_nodes = []  # type: List[BaseNode]
        self.gradients = dict()  # type: Dict[BaseNode, float]
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        """Forward the input of inbound_nodes."""
        raise NotImplementedError

    def backward(self):
        """Backard the gradient of outbound_nodes."""
        raise NotImplementedError
