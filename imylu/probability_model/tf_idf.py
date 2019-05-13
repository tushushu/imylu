"""
@Author: tushushu
@Date: 2019-05-09 13:30:58
"""
import numpy as np
from numpy import array
from itertools import chain
from collections import Counter
from typing import List, Dict


class TFIDF:
    def init(self):
        self.tf = None
        self.idf = None
        self.word2num = None
        self.num2word = None

    @staticmethod
    def get_word2num(data: List[str]) -> Dict[str, int]:
        """[summary]

        Arguments:
            data {List[str]} -- [description]

        Returns:
            Dict[str, int] -- [description]
        """

        words = set(chain(*data))
        return {word: i for i, word in enumerate(words)}

    @staticmethod
    def get_num2word(word2num: dict) -> Dict[str, int]:
        return {word: i for i, word in word2num.items()}

    @staticmethod
    def get_word_cnt(data: array) -> Counter:
        """[summary]

        Arguments:
            data {array} -- [description]

        Returns:
            Counter -- [description]
        """

        return Counter(data.flatten())

    def fit(self, data: array):
        raise NotImplementedError
