"""
@Author: tushushu
@Date: 2019-05-21 20:11:23
"""
import numpy as np
from numpy import array
from typing import Callable


class Dense:
    def __init__(self, input_size: int, output_size: int, activator: Callable):
        """[summary]

        Arguments:
            input_size {int} -- [description]
            output_size {int} -- [description]
            activator {Callable} -- [description]
        """

        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        self.weights = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.bias = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array: array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        # 式2
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.weights, input_array) + self.bias)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.weights.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.weights += learning_rate * self.W_grad
        self.bias += learning_rate * self.b_grad

