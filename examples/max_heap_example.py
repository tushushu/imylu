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
from time import time


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


def exhausted_search(nums, k):
    """Linear search the top k smallest elements.
    """

    rets = []
    idxs = []
    key = None
    val = float("inf")
    for _ in range(k):
        for i, num in enumerate(nums):
            if num < val:
                flag = True
                for idx in idxs:
                    if idx == i:
                        flag = False
                        break
                if flag:
                    key = i
                    val = num
        idxs.append(key)
        rets.append(val)
        val = float("inf")
    return rets


def main():
    # Test
    print("Testing MaxHeap...")
    test_times = 100
    run_time_1 = run_time_2 = 0
    for _ in range(test_times):
        # Generate dataset randomly
        low = 0
        high = 1000
        n_rows = 10000
        k = 100
        nums = gen_data(low, high, n_rows)

        # Build Max Heap
        heap = MaxHeap(k, lambda x: x)
        start = time()
        for num in nums:
            heap.add(num)
        ret1 = heap.items
        run_time_1 += time() - start

        # Exhausted search
        start = time()
        ret2 = exhausted_search(nums, k)
        run_time_2 += time() - start

        # Compare result
        ret1.sort()
        assert ret1 == ret2, "target:%s\nk:%d\nrestult1:%s\nrestult2:%s\n" % (
            str(nums), k, str(ret1), str(ret2))
    print("%d tests passed!" % test_times)
    print("Max Heap Search %.2f s" % run_time_1)
    print("Exhausted search %.2f s" % run_time_2)


if __name__ == "__main__":
    main()
