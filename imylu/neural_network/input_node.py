"""
@Author: tushushu
@Date: 2019-05-30 14:38:56
"""
from .base_node import BaseNode


class InputNode(BaseNode):
    """[summary]

    Arguments:
        BaseNode {[type]} -- [description]
    """

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            self.gradients[self] += node.gradients[self]
