# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-13 17:15:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-13 17:15:29
"""


class KNeighborsBase(object):
    def __init__(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def _predict(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
