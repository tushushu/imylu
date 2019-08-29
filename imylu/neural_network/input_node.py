"""
@Author: tushushu
@Date: 2019-05-30 14:38:56
"""
from .base_node import BaseNode


class InputNode(BaseNode):
    """Input node class.

    Arguments:
        value {Optional[float]} -- The value of Node.
        inbound_nodes {List[Node]]} -- inbound nodes.
        outbound_nodes {List[Node]} -- outbound nodes.
        gradients {Dict[Node, float]} -- Keys: inbound nodes, Values: gradients.
    """

    def forward(self):
        """Forward the input of inbound_nodes."""
        pass

    def backward(self):
        """Backard the gradient of outbound_nodes."""
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            self.gradients[self] += node.gradients[self]
