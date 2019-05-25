# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2019-05-25 20:18:14
"""
import numpy as np
from numpy import array


class FullyConnectedLayer:
    """[summary]
    """

    def __init__(self, n_nodes: int, n_inputs: int, next_layer):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.next_layer = next_layer
        self.weights = self.init_weights()
        self.inputs = None
        self.inputs_extended = None

    def init_weights(self)->array:
        """[summary]

        Returns:
            array -- [description]
        """

        size = (self.n_nodes + 1, self.n_inputs)
        return np.random.normal(size=size)

    @staticmethod
    def extend_inputs(inputs: array)->array:
        """[summary]

        Arguments:
            inputs {array} -- [description]

        Returns:
            array -- [description]
        """

        bias_size = 1 if inputs.dim == 1 else inputs.shape[1]
        bias = np.ones(shape=(bias_size, None))
        return np.append(inputs, bias, axis=1)

    def forward(self):
        """[summary]
        """

        # Inputs inspection.
        inputs = self.inputs
        assert isinstance(inputs, array), "Inputs has to be a numpy array!"
        assert inputs.shape[0] == self.n_inputs, "Except a (%d, any) array, but got a %s one!" % (
            self.n_inputs, inputs.shape)

        # Extend inputs with ones.
        self.inputs_extended = self.extend_inputs(inputs)

        # Calculate and forward outputs.
        outputs = np.matmul(self.weights, self.inputs_extended)
        self.next_layer.inputs = outputs

    def backward(self):
        """[summary]
        """
        raise NotImplementedError
