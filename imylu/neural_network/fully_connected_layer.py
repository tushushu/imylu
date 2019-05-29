# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2019-05-25 20:18:14
"""
import numpy as np
from numpy import array
from .base_layer import BaseLayer


class FullyConnectedLayer(BaseLayer):
    """[summary]
    """

    def __init__(self, name: str, input_size: int, output_size: int, activate_fn=None):
        super(FullyConnectedLayer, self).__init__(name, input_size, output_size)
        self.activate_fn = activate_fn
        self.weights = self.init_weights()

    def init_weights(self) -> array:
        """[summary]

        Returns:
            array -- [description]
        """

        size = (self.output_size, self.input_size)
        return np.random.normal(size=size)

    def forward(self):
        """[summary]
        """

        # Calculate outputs.
        inputs = np.concatenate(x for x in self.inputs.values())
        outputs = np.matmul(self.weights, inputs)
        if self.activate_fn:
            outputs = self.activate_fn(outputs)

        # Forward outputs.
        self.output_layer.inputs[self.name] = outputs

    def backward(self):
        """[summary]
        """
        raise NotImplementedError
