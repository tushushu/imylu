"""
@Author: tushushu
@Date: 2019-05-29 15:31:55
"""

class BaseNetwork:
    def __init__(self):
        # Layer size inspection.
        # input_layers_output_size = sum(layer.output_size for layer in self.input_layers.values())
        # err_msg = "Input layers' output size does not match current layer's input size!"
        # assert input_layers_output_size == self.input_size, err_msg