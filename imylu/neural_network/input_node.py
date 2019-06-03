"""
@Author: tushushu
@Date: 2019-05-30 14:38:56
"""
from .base_node import Node

class InputNode(Node):
    """[summary]

    Arguments:
        Node {[type]} -- [description]
    """
    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for outbound_node in self.outbound_nodes.values():
            self.gradients[self] += outbound_node.gradients[self.name]
