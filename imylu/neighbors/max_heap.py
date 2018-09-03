# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-09-03 15:07:15
@Last Modified by:   tushushu
@Last Modified time: 2018-09-03 15:07:15
"""


class MaxHeap(object):
    def __init__(self, max_size, fn):
        """MaxHeap class.

        Arguments:
            max_size {int} -- The maximum size of MaxHeap instance.
            fn {function} -- Function to caculate the values of items when comparing.

        Attributes:
            items {object} -- The items in the MaxHeap instance.
            size {int} -- The size of MaxHeap instance.
        """

        self.max_size = max_size
        self.fn = fn

        self.items = [None] * max_size
        self.size = 0

    def __str__(self):
        item_values = str(map(self.fn, self.items))
        return "Size: %d\nMax size: %d\nItem_values: %s\n" % (self.size, self.max_size, item_values)

    def value(self, idx):
        """Caculate the value of item.

        Arguments:
            idx {int} -- The index of item.

        Returns:
            float
        """

        return self.fn(self.items[idx])

    def add(self, item):
        """Add a new item to the MaxHeap.

        Arguments:
            item {object} -- The item to add.
        """

        assert self.size < self.max_size, "Cannot add item! The MaxHeap is full!"
        self.items[self.size] = item
        self.size += 1
        self._shift_up(self.size - 1)

    def pop(self):
        """Pop the top item out of the heap.

        Returns:
            object -- The item popped.
        """

        assert self.size > 0, "Cannot pop item! The MaxHeap is empty!"
        self.items[0] = self.items[self.size - 1]
        self.size -= 1
        self._shift_down(0)
        return self.items[0]

    def _shift_up(self, idx):
        """Shift up item unitl its parent is greater than the item.

        Arguments:
            idx {int} -- Heap item's index.
        """

        parent = (idx - 1) // 2
        while parent > 0 and self.value(parent) < self.value(idx):
            self.items[parent], self.items[idx] = self.items[idx], self.items[parent]
            idx, parent = parent, (idx - 1) // 2

    def _shift_down(self, idx):
        """Shift down item until its children are less than the item.

        Arguments:
            idx {int} -- Heap item's index.
        """

        child = (idx + 1) * 2 - 1
        while child < self.size and self.value(idx) < self.value(child):
            # Compare the left child and the right child and get the index of the larger one.
            if child + 1 < self.size and self.value(child + 1) > self.value(child):
                child += 1
            self.items[idx], self.items[child] = self.items[child], self.items[idx]
            idx, child = child, (idx + 1) * 2 - 1
