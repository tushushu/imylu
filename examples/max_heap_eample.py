# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-09-03 17:34:41
@Last Modified by:   tushushu
@Last Modified time: 2018-09-03 17:34:41
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append(os.path.abspath(".."))

from imylu.neighbors.max_heap import MaxHeap
from imylu.utils import gen_data
from random import randint


def is_valid(heap):
    """Validate a MaxHeap by comparing all the parents and its children.

    Arguments:
        heap {MaxHeap}

    Returns:
        bool
    """

    ret = []
    for i in range(1, heap.size):
        parent = (i - 1) // 2
        ret.append(heap.value(parent) >= heap.value(i))
        if heap.value(parent) < heap.value(i):
            print(parent, i, heap.value(parent), heap.value(i))
    return all(ret)


def main():
    print("Testing MaxHeap...")
    test_times = 1000
    for _ in range(test_times):
        max_size = randint(1, 100)
        heap = MaxHeap(max_size, lambda x: x[0])
        X = gen_data(low=0, high=100, n_rows=max_size, n_cols=3)

        validations = []
        for Xi in X:
            heap.add(Xi)
            validations.append(is_valid(heap))
        assert all(validations), "Test failed!"

        validations = []
        for _ in range(heap.size):
            heap.pop()
            validations.append(is_valid(heap))
        assert all(validations), "Test failed! %s" % str(heap)

    print("%d tests passed!" % test_times)
    heap = MaxHeap(18, lambda x: x[0])


if __name__ == "__main__":
    main()
