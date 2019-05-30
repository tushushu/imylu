"""
@Author: tushushu
@Date: 2019-05-29 15:23:35
"""

from typing import Dict


class Node:
    """Support multiple inputs and multiple outputs.
    """

    def __init__(self, name: str):
        self.name = name
        self.inbound_nodes = dict()  # type: Dict[str, Node]
        self.outbound_nodes = dict()  # type: Dict[str, Node]
        self.gradients = dict()  # type: Dict[str, Node]

    @property
    def inbound_node_names(self):
        """[summary]

        Returns:
            [type] -- [description]
        """

        return self.inbound_nodes.keys()

    @property
    def outbound_node_names(self):
        """[summary]

        Returns:
            [type] -- [description]
        """

        return self.outbound_nodes.keys()

    def add(self, node: Node):
        """[summary]

        Arguments:
            node {Node} -- [description]
        """

        # Current node name inspection.
        err_msg = "Node name %s already in use!" % self.name
        assert self.name not in node.inbound_node_names, err_msg

        # Outbound node name inspection.
        err_msg = "Node name %s already in use!" % node.name
        assert node.name not in self.outbound_node_names, err_msg

        # Connect current node and outbound node.
        self.outbound_nodes[node.name] = node
        node.inbound_nodes[self.name] = self

    def forward(self):
        """[summary]
        """

        raise NotImplementedError

    def backward(self):
        """[summary]
        """
        raise NotImplementedError
