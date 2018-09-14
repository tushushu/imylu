
提到大顶堆相信大家应该都不会觉得陌生（不陌生你点进来干嘛[捂脸]），大名鼎鼎的KNN算法就用到了大顶堆。本文就大顶堆的基本原理进行讲解，并手把手、肩并肩地带您实现这一算法。

完整实现代码请参考本人的p...哦不是...github：
[max_heap.py](https://github.com/tushushu/imylu/blob/master/imylu/neighbors/max_heap.py)
[max_heap_example.py](https://github.com/tushushu/imylu/blob/master/examples/max_heap_example.py)


# 1. 原理篇
我们用大白话讲讲大顶堆是怎么一回事。

## 1.1 什么是“堆”
在实际生活中，“堆”非常常见，比如工地旁边会有“土堆”，一些垃圾站会有“垃圾堆”。这些“堆”通常都是由一些相似的物体放在一起，形成上窄下宽的结构。

## 1.2 什么是“大顶堆”
如下图所示，计算机中的“堆”就是把数据放在一颗完全二叉树中，形状看上去跟我们提到的“土堆”，“垃圾堆”差不多。跟普通二叉树的区别就是，每个父节点的值都大于子节点的值，“富不过三代”用大顶堆来描述再贴切不过。

![gbdt_classifier.png](https://github.com/tushushu/imylu/blob/master/pic/gbdt_classifier.png)


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

## 2.5 删除顶部元素
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
    parent = (idx - 1) // 2
    while parent >= 0 and self.value(parent) < self.value(idx):
        self.items[parent], self.items[idx] = self.items[idx], self.items[parent]
        idx = parent
        parent = (idx - 1) // 2
```

## 2.6 元素下沉
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
## 3.1 heap校验
```Python
def is_valid(heap):
    ret = []
    for i in range(1, heap.size):
        parent = (i - 1) // 2
        ret.append(heap.value(parent) >= heap.value(i))
    return all(ret)
```

## 3.2 线性查找
```Python
def exhausted_search(nums, k):
    rets = []
    idxs = []
    key = None
    val = float("inf")
    for _ in range(k):
        for i, num in enumerate(nums):
            if num < val and i not in idxs:
                key = i
                val = num
        idxs.append(key)
        rets.append(val)
        val = float("inf")
    return rets
```

## 3.3 main函数
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

        # Test pop method
        while heap.size > 0:
            heap.pop()
            assert is_valid(heap), "Invalid heap!"

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
最终准确度93.082%，运行时间14.9秒，效果还算不错~
![gbdt_classifier.png](https://github.com/tushushu/imylu/blob/master/pic/gbdt_classifier.png)

## 3.3 工具函数
本人自定义了一些工具函数，可以在github上查看
[utils.py](https://github.com/tushushu/imylu/blob/master/imylu/utils.py)
1. run_time - 测试函数运行时间
2. load_breast_cancer - 加载乳腺癌数据
3. train_test_split - 拆分训练集、测试集
4. get_acc - 计算准确度


# 总结
GBDT分类的原理：GBDT回归加Sigmoid

GBDT分类的实现：一言难尽[哭]

