"""
@Author: tushushu
@Date: 2019-05-29 15:23:35
"""

from numpy import array
from typing import Dict, Optional


class BaseLayer:
    """Support multiple inputs and single output.
    """

    def __init__(self, name: str, input_size: int, output_size: int):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size

        self.inputs = dict()  # type: Dict[str, array]
        self.output = None

        self.input_layers = dict()  # type: Dict[str, BaseLayer]
        self.output_layer = None  # type: Optional[BaseLayer]

    def add(self, layer: BaseLayer):
        """[summary]

        Arguments:
            layer {BaseLayer} -- [description]
        """

        # Layer name inspection.
        assert self.name not in layer.input_layers, "Layer name %s already in use!" % self.name

        self.output_layer = layer
        layer.input_layers[self.name] = self

    def forward(self):
        """[summary]
        """

        raise NotImplementedError

    def backward(self):
        """[summary]
        """
        raise NotImplementedError
