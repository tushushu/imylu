
提到大顶堆相信大家应该都不会觉得陌生（不陌生你点进来干嘛[捂脸]），大名鼎鼎的KNN算法就用到了大顶堆。本文就大顶堆的基本原理进行讲解，并手把手、肩并肩地带您实现这一算法。

完整实现代码请参考本人的p...哦不是...github：  
[max_heap.py](https://github.com/tushushu/imylu/blob/master/imylu/utils/max_heap.py)  
[max_heap_example.py](https://github.com/tushushu/imylu/blob/master/examples/max_heap_example.py)  


# 1. 原理篇
我们用大白话讲讲大顶堆是怎么一回事。

## 1.1 什么是“堆”
在实际生活中，“堆”非常常见，比如工地旁边会有“土堆”，一些垃圾站会有“垃圾堆”。这些“堆”通常都是由一些相似的物体放在一起，形成上窄下宽的结构。

## 1.2 完全二叉树
百度百科说：对于深度为K的，有n个节点的二叉树，当且仅当其每一个节点都与深度为K的满二叉树中编号从1至n的节点一一对应时称之为完全二叉树。
这描述让人听起来有点懵逼，说得简单点，完全二叉树只有两种情况：  
1. 完美二叉树，即每一层节点数都是上一层的二倍
2. 完美二叉树扣掉若干个节点，"扣"的顺序是由下向上、由右向左

## 1.2 什么是“大顶堆”
如下图所示，计算机中的“大顶堆”就是把数据放在一颗完全二叉树中，形状看上去跟我们提到的“土堆”，“垃圾堆”差不多。跟普通二叉树的区别就是，每个父节点的值都大于子节点的值，即儿子不如爹，所以用大顶堆来描述“富不过三代”再贴切不过。  
![max_heap.png](https://github.com/tushushu/imylu/blob/master/pic/max_heap.png)

## 1.3 如何建立大顶堆
建立一个大顶堆需要告诉计算机：这，就是大顶堆！然后要说明这个大顶堆目前的大小是0，未来不能超过多大。由于大顶堆是个完全二叉树，层序遍历的时候元素都是连续的，中间没有“空位”，所以很方便用数组来表示这棵树。那么我们就再开辟一个数组，用于存储大顶堆的元素。

## 1.4 元素上浮
之前说过大顶堆的特征是“儿子不如爹”，那么如果大顶堆的最后一个元素比爹还大，那么这个儿子就要升级当爹了，爹也要降级为儿子，听起来有点乱...这个过程就是元素的上浮过程。如果上浮一次之后，发现儿子还是比爹大，就继续上浮，直到上浮到爹比儿子大或者上浮到堆顶为止。

## 1.5 元素下沉
同理，如果大顶堆的第一个元素比儿子还小，那么这个爹就要降级为儿子了，儿子也要升级当爹，听起来仍然有点乱...这个过程就是元素的下沉过程。如果下沉一次之后，发现爹还是比儿子小，就继续下沉，直到下沉到儿子比爹小或者下沉到堆底为止。

## 1.6 插入元素
在大顶堆中插入一个元素，分为如下两种情况：  
1. 堆未满，将元素放在当前最后一个元素的后面，然后执行上浮过程；
2. 堆已满，如果该元素大于堆顶则无法插入，小于堆顶则替换堆顶，再执行下沉过程。

## 1.7 推出顶部元素
大顶堆的交换顶部元素A和最后一个元素B，堆的size减1，再将顶部的B执行下沉过程，最后返回元素A。注意，虽然堆的size减小了1，但实际上并没有元素被删除，数组长度也没有任何变化，被pop的元素只是被放在了数组中size之后的位置。

## 1.8 大顶堆有什么用
大顶堆的典型应用有3个：
1. 堆排序降序，我们把顶点元素不停地pop出来，由于每次pop出的元素都是当时最大的，所以把pop的值收集起来就是一个降序数组；
2. 堆排序升序，同方法1，由于顶点元素每次都被pop方法放在了数组的最后一个元素的位置，所以全部pop完毕之后堆中的数组已经是一个升序数组；
3. 从N个元素中查找最小的K个元素，把N个元素逐个插入大小为K的大顶堆中，最后大顶堆中的元素就是我们要找的TOP K。

# 2. 实现篇
本人用全宇宙最简单的编程语言——Python实现了大顶堆算法，没有依赖任何第三方库，便于学习和使用。简单说明一下实现过程，更详细的注释请参考本人github上的代码。

## 2.1 创建MaxHeap类
初始化，存储最大元素数量、元素值计算函数、元素列表，当前元素数量。
```Python
class MaxHeap(object):
    def __init__(self, max_size, fn):
        self.max_size = max_size
        self.fn = fn
        self.items = [None] * max_size
        self.size = 0
```

## 2.2 获取大顶堆各个属性
```Python
def __str__(self):
    item_values = str([self.fn(self.items[i]) for i in range(self.size)])
    return "Size: %d\nMax size: %d\nItem_values: %s\n" % (self.size, self.max_size, item_values)
```

## 2.3 检查大顶堆是否已满
```Python
@property
def full(self):
    return self.size == self.max_size
```

## 2.4 添加元素
```Python
def add(self, item):
    if self.full:
        if self.fn(item) < self.value(0):
            self.items[0] = item
            self._shift_down(0)
    else:
        self.items[self.size] = item
        self.size += 1
        self._shift_up(self.size - 1)
```

## 2.5 推出顶部元素
```Python
def pop(self):
    assert self.size > 0, "Cannot pop item! The MaxHeap is empty!"
    ret = self.items[0]
    self.items[0] = self.items[self.size - 1]
    self.items[self.size - 1] = None
    self.size -= 1
    self._shift_down(0)
    return ret
```

## 2.6 元素上浮
```Python
def _shift_up(self, idx):
    assert idx < self.size, "The parameter idx must be less than heap's size!"
    parent = (idx - 1) // 2
    while parent >= 0 and self.value(parent) < self.value(idx):
        self.items[parent], self.items[idx] = self.items[idx], self.items[parent]
        idx = parent
        parent = (idx - 1) // 2
```

## 2.7 元素下沉
```Python
def _shift_down(self, idx):
    child = (idx + 1) * 2 - 1
    while child < self.size:
        if child + 1 < self.size and self.value(child + 1) > self.value(child):
            child += 1
        if self.value(idx) < self.value(child):
            self.items[idx], self.items[child] = self.items[child], self.items[idx]
            idx = child
            child = (idx + 1) * 2 - 1
        else:
            break
```


# 3 效果评估
## 3.1 heap有效性校验
检查当前堆是否符合大顶堆的定义。
```Python
def is_valid(heap):
    ret = []
    for i in range(1, heap.size):
        parent = (i - 1) // 2
        ret.append(heap.value(parent) >= heap.value(i))
    return all(ret)
```

## 3.2 线性查找
用“笨”办法查找最小的k个元素。
```Python
def exhausted_search(nums, k):
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
```

## 3.3 main函数
主函数分为如下几个部分：
1. 随机生成数据集，即测试用例
2. 建立大顶堆
3. 执行“笨”办法查找
4. 比较“笨”办法和大顶堆的查找结果
```Python
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
        ret1 = copy(heap.items)
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
```
## 3.2 效果展示
针对top k查找随机生成了100个测试用例，线性查找用时8.76秒，大顶堆用时1.74秒，效果还算不错~
![max_heap1.png](https://github.com/tushushu/imylu/blob/master/pic/max_heap1.png)


# 总结
我都写这么清楚了，你还忍心让我总结？[捂脸]

