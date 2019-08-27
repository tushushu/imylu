"""
@Author: tushushu
@Date: 2019-05-30 14:44:33
"""
import numpy as np
from .base_node import BaseNode


class Sigmoid(BaseNode):
    """[summary]
    """
    def __init__(self, input_node):
        BaseNode.__init__(self, input_node)

    @staticmethod
    def _sigmoid(arr):
        return 1. / (1. + np.exp(-arr))

    @staticmethod
    def _derivative(arr):
        return arr * (1 - arr)

    def forward(self):
        input_node = self.inbound_nodes[0]
        self.value = self._sigmoid(input_node.value)

    def backward(self):
        input_node = self.inbound_nodes[0]
        self.gradients = {input_node: np.zeros_like(input_node.value)}
        for output_node in self.outbound_nodes:
            grad_cost = output_node.gradients[self]
            self.gradients[input_node] += self._derivative(self.value) * grad_cost
