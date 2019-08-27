# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2019-05-25 20:18:14
"""
import numpy as np
from .base_node import BaseNode
from .input_node import InputNode


class Linear(BaseNode):
    """[summary]

    Arguments:
        BaseNode {[type]} -- [description]
    """
    def __init__(self, data: InputNode, weights: InputNode, bias: InputNode):
        BaseNode.__init__(self, data, weights, bias)

    def forward(self):
        data, weights, bias = self.inbound_nodes
        self.value = np.dot(data.value, weights.value) + bias.value

    def backward(self):
        data, weights, bias = self.inbound_nodes
        self.gradients = {node: np.zeros_like(node.value) for node in self.inbound_nodes}
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            self.gradients[data] += np.dot(grad_cost, weights.value.T)
            self.gradients[weights] += np.dot(data.value.T, grad_cost)
            self.gradients[bias] += np.sum(grad_cost, axis=0, keepdims=False)
