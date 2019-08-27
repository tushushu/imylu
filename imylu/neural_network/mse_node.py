"""
@Author: tushushu
@Date: 2019-05-30 14:46:27
"""
import numpy as np
from .base_node import BaseNode
from .input_node import InputNode


class MSE(BaseNode):
    def __init__(self, label: InputNode, pred: BaseNode):
        BaseNode.__init__(self, label, pred)
        self.m = None
        self.diff = None

    def forward(self):
        label, pred = self.inbound_nodes
        self.m = label.value.shape[0]
        self.diff = (label.value - pred.value).reshape(-1, 1)
        self.value = np.mean(self.diff**2)

    def backward(self):
        label, pred = self.inbound_nodes
        self.gradients[label] = (2 / self.m) * self.diff
        self.gradients[pred] = -self.gradients[label]
