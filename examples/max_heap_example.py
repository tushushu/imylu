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

from copy import copy
from imylu.utils.load_data import gen_data
from imylu.utils.max_heap import MaxHeap
from time import time


def exhausted_search(nums, k):
    """Linear search the top k smallest elements.

    Arguments:
        nums {list} -- 1d list with int or float.
        k {int}

    Returns:
        list -- Top k smallest elements.
    """

    rets = []
    idxs = []
    key = None
    for _ in range(k):
        val = float("inf")
        for i, num in enumerate(nums):
            if num < val and i not in idxs:
                key = i
                val = num
        idxs.append(key)
        rets.append(val)

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
        ret1 = copy(heap._items)
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
