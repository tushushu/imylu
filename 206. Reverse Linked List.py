# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-11-06 11:03:20
@Last Modified by:   tushushu
@Last Modified time: 2018-11-06 11:03:20
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        stack = [None]
        node = head
        while node:
            stack.append(node)
            node = node.next
        ret = stack.pop()
        node_cur = ret
        while stack:
            node_next = stack.pop()
            node_cur.next = node_next
            node_cur = node_next
        return ret
