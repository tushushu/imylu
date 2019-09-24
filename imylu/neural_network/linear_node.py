# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2019-05-25 20:18:14
"""
import numpy as np
from .base_node import BaseNode
from .weight_node import WeightNode


class LinearNode(BaseNode):
    """Linear node class.

    Attributes:
        name {str} -- The name of Node.
        value {Optional[float]} -- The value of Node.
        inbound_nodes {List[Node]]} -- inbound nodes.`
        outbound_nodes {List[Node]} -- outbound nodes.
        gradients {Dict[Node, float]} -- Keys: inbound nodes, Values: gradients.
    """
    def __init__(self, data: BaseNode, weights: WeightNode, bias: WeightNode, name=None):
        """Initialize a node instance and connect inbound nodes to this instance.

        Arguments:
            data {BaseNode} -- The value of data is an ndarray with shape(m, n).
            weights {WeightNode} -- The value of weights is an ndarray with shape(n, k).
            bias {WeightNode} -- The value of bias is an ndarray with shape(1, k).
        """
        BaseNode.__init__(self, data, weights, bias, name=name)

    def forward(self):
        """Forward the input of inbound_nodes.

        Y = WX + Bias
        """
        data, weights, bias = self.inbound_nodes
        self.value = np.dot(data.value, weights.value) + bias.value

    def backward(self):
        """Backward the gradient of outbound_nodes.

        dY / dX = W
        dY / dW = X
        dY / dBias = 1
        """
        data, weights, bias = self.inbound_nodes
        self.gradients = {node: np.zeros_like(node.value) for node in self.inbound_nodes}
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            self.gradients[data] += np.dot(grad_cost, weights.value.T)
            self.gradients[weights] += np.dot(data.value.T, grad_cost)
            self.gradients[bias] += np.sum(grad_cost, axis=0, keepdims=False)
