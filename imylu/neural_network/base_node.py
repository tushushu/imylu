"""
@Author: tushushu
@Date: 2019-05-29 15:23:35
"""

from typing import List, Dict


class BaseNode:
    """
    BaseNode class.

    Arguments:
        value {Optional[float]} -- The value of BaseNode.
        inbound_nodes {List[BaseNode]]} -- inbound nodes.
        outbound_nodes {List[BaseNode]} -- outbound nodes.
        gradients {Dict[BaseNode, float]} -- Keys: inbound nodes, Values: gradients.
    """
    def __init__(self, *inbound_nodes):
        self.value = None
        self.inbound_nodes = list(inbound_nodes) if inbound_nodes else []  # type: List[BaseNode]
        self.outbound_nodes = []  # type: List[BaseNode]
        self.gradients = dict()  # type: Dict[BaseNode, float]
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        """[summary]
        """
        raise NotImplementedError

    def backward(self):
        """[summary]
        """
        raise NotImplementedError
