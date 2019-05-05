
提到回归树相信大家应该都不会觉得陌生（不陌生你点进来干嘛[捂脸]），大名鼎鼎的GBDT算法就是用回归树组合而成的。本文就回归树的基本原理进行讲解，并手把手、肩并肩地带您实现这一算法。


完整实现代码请参考本人的p...哦不是...github：  
[regression_tree.py](https://github.com/tushushu/imylu/blob/master/imylu/tree/regression_tree.py)  
[regression_tree_example.py](https://github.com/tushushu/imylu/blob/master/examples/regression_tree_example.py)  


# 1. 原理篇
我们用人话而不是大段的数学公式来讲讲回归树是怎么一回事。
## 1.1 最简单的模型
如果预测某个连续变量的大小，最简单的模型之一就是用平均值。比如同事的平均年龄是28岁，那么新来了一批同事，在不知道这些同事的任何信息的情况下，直觉上用平均值28来预测是比较准确的，至少比0岁或者100岁要靠谱一些。我们不妨证明一下我们的直觉：

1. 定义损失函数L，其中y_hat是对y预测值，使用MSE来评估损失：  
$L = -\Large\frac{1}{2}\normalsize\sum_{i=0}^m(y_i-\hat{y}) ^ 2$

2. 对y_hat求导:  
$
\Large
\frac{\mathrm{d}L}{\mathrm{d}\hat{y}}
\normalsize 
= \sum_{i=0}^m(y_i-\hat{y})
= \sum_{i=0}^my_i - \sum_{i=0}^m\hat{y}
= \sum_{i=0}^my_i - m*\hat{y}
$  

3. 令导数等于0，最小化MSE，则:  
$\sum_{i=0}^my_i - m*\hat{y} = 0$   

4. 所以，  
$
\hat{y} 
= \Large\frac{1}{m}\normalsize\sum_{i=0}^my_i 
$  

结论，如果要用一个常量来预测y，用y的均值是一个最佳的选择。

## 1.2 加一点难度
仍然是预测同事年龄，这次我们预先知道了同事的职级，假设职级的范围是整数1-10，如何能让这个信息帮助我们更加准确的预测年龄呢？

一个思路是根据职级把同事分为两组，这两组分别应用我们之前提到的“平均值”模型。比如职级小于5的同事分到A组，大于或等于5的分到B组，A组的平均年龄是25岁，B组的平均年龄是35岁。如果新来了一个同事，职级是3，应该被分到A组，我们就预测他的年龄是25岁。

## 1.3 最佳分割点
还有一个问题待解决，如何取一个最佳的分割点对不同职级的同事进行分组呢？
我们尝试所有m个可能的分割点P_i，沿用之前的损失函数，计算Loss得到L_i。最小的L_i所对应的P_i就是我们要找的“最佳分割点”。

## 1.4 运用多个变量
再复杂一些，如果我们不仅仅知道了同事的职级，还知道了同事的工资（貌似不科学），该如何预测同事的年龄呢？

我们可以分别根据职级、工资计算出职级和工资的最佳分割点P_1, P_2，对应的Loss L_1, L_2。然后比较L_1和L2，取较小者。假设L_1 < L_2，那么按照P_1把不同职级的同事分为A、B两组。在A、B组内分别计算工资所对应的分割点，再分为C、D两组。这样我们就得到了AC, AD, BC, BD四组同事以及对应的平均年龄用于预测。

## 1.5 答案揭晓
如何实现这种1 to 2, 2 to 4, 4 to 8的算法呢？

熟悉数据结构的同学自然会想到二叉树，这种树被称为回归树，顾名思义利用树形结构求解回归问题。

# 2. 实现篇
本人用全宇宙最简单的编程语言——Python实现了回归树算法，便于学习和使用。简单说明一下实现过程，更详细的注释请参考本人github上的代码。
## 2.1 创建Node类
初始化，存储预测值、左右结点、特征和分割点
```Python
class Node:
    def __init__(self, score=None):
        self.score = score
        self.left = None
        self.right = None
        self.feature = None
        self.split = None
```

## 2.2 创建回归树类
初始化，存储根节点和树的高度。
```Python
class RegressionTree(object):
    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None
```

## 2.3 计算分割点、MSE
根据自变量col、因变量label以及分割点split，计算分割后的MSE。
```Python
@staticmethod
def _get_split_mse(col: array, score: array, split: float):
    # Split score.
    score_left = score[col < split]
    score_right = score[col >= split]

    # Calculate the means of score.
    avg_left = score_left.mean()
    avg_right = score_right.mean()

    # Calculate the mse of score.
    mse = (((score_left - avg_left) ** 2).sum() +
            ((score_right - avg_right) ** 2).sum()) / len(score)

    return mse, avg_left, avg_right
```

## 2.4 计算最佳分割点
遍历特征某一列的所有的不重复的点，找出MSE最小的点作为最佳分割点。如果特征中没有不重复的元素则返回None。
```Python
def _choose_split(self, col: array, score: array):
    # Feature cannot be splitted if there's only one unique element.
    unique = set(col)
    if len(unique) == 1:
        return None, None, None, None
    # In case of empty split
    unique.remove(min(unique))

    # Get split point which has min mse
    ite = map(lambda x: (*self._get_split_mse(col, score, x), x), unique)
    mse, avg_left, avg_right, split = min(ite, key=lambda x: x[0])

    return mse, avg_left, avg_right, split
```

## 2.5 选择最佳特征
遍历所有特征，计算最佳分割点对应的MSE，找出MSE最小的特征、对应的分割点，左右子节点对应的均值。如果所有的特征都没有不重复元素则返回None
```Python
def _choose_feature(self, data: array, score: array):
    # Compare the mse of each feature and choose best one.
    ite = map(lambda x: (*self._choose_split(
        data[:, x], score), x), range(data.shape[1]))
    ite = filter(lambda x: x[0] is not None, ite)

    # Terminate if no feature can be splitted
    return min(ite, default=None, key=lambda x: x[0])
```

## 2.6 规则转文字
将规则用文字表达出来，方便我们查看规则。
```Python
@staticmethod
def _expr2literal(expr):
    feature, operation, split = expr
    operation = ">=" if operation == 1 else "<"
    return "Feature%d %s %.4f" % (feature, operation, split)
```

## 2.7 获取规则
将回归树的所有规则都用文字表达出来，方便我们了解树的全貌。这里用到了队列+广度优先搜索。有兴趣也可以试试递归或者深度优先搜索。
```Python
def _get_rules(self):
    que = [[self.root, []]]
    self._rules = []
    # Breadth-First Search
    while que:
        node, exprs = que.pop(0)
        # Generate a rule when the current node is leaf node
        if not(node.left or node.right):
            # Convert expression to text
            literals = list(map(self._expr2literal, exprs))
            self._rules.append([literals, node.score])
        # Expand when the current node has left child
        if node.left:
            rule_left = copy(exprs)
            rule_left.append([node.feature, -1, node.split])
            que.append([node.left, rule_left])
        # Expand when the current node has right child
        if node.right:
            rule_right = copy(exprs)
            rule_right.append([node.feature, 1, node.split])
            que.append([node.right, rule_right])
```

## 2.8 训练模型
仍然使用队列+广度优先搜索，训练模型的过程中需要注意：
1. 控制树的最大深度max_depth；
2. 控制分裂时最少的样本量min_samples_split；
3. 叶子结点至少有两个不重复的y值；
4. 至少有一个特征是没有重复值的。
```Python
def fit(self, data: array, score: array, max_depth=5, min_samples_split=2):
    # Initialize with depth, node, indexes
    self.root.score = score.mean()
    que = [(self.depth + 1, self.root, data, score)]
    # Breadth-First Search
    while que:
        depth, node, _data, _score = que.pop(0)
        # Terminate loop if tree depth is more than max_depth
        if depth > max_depth:
            depth -= 1
            break
        # Stop split when number of node samples is less than
        # min_samples_split or Node is 100% pure.
        if len(_score) < min_samples_split or all(_score == score[0]):
            continue
        # Stop split if no feature has more than 2 unique elements
        split_ret = self._choose_feature(_data, _score)
        if split_ret is None:
            continue
        # Split
        _, avg_left, avg_right, split, feature = split_ret
        # Update properties of current node
        node.feature = feature
        node.split = split
        node.left = Node(avg_left)
        node.right = Node(avg_right)
        # Put children of current node in que
        idx_left = (_data[:, feature] < split)
        idx_right = (_data[:, feature] >= split)
        que.append(
            (depth + 1, node.left, deepcopy(_data[idx_left]), deepcopy(_score[idx_left])))
        que.append(
            (depth + 1, node.right, deepcopy(_data[idx_right]), deepcopy(_score[idx_right])))
    # Update tree depth and rules
    self.depth = depth
    self._get_rules()

```
## 2.9 打印规则
模型训练完毕，查看一下模型生成的规则
```Python
    def __str__(self):
        ret = []
        for i, rule in enumerate(self._rules):
            literals, score = rule

            ret.append("Rule %d: " % i + ' | '.join(
                literals) + ' => y_hat %.4f' % score)
        return "\n".join(ret)
```

## 2.10 预测一个样本
```Python
def _predict(self, row: array)->float:
    node = self.root
    while node.left and node.right:
        if row[node.feature] < node.split:
            node = node.left
        else:
            node = node.right
    return node.score
```

## 2.11 预测多个样本
```Python
def predict(self, data: array)->array:
    return np.apply_along_axis(self._predict, 1, data)
```

# 3 效果评估
## 3.1 main函数
使用著名的波士顿房价数据集，按照7:3的比例拆分为训练集和测试集，训练模型，并统计准确度。
```Python
def main():
    print("Tesing the performance of RegressionTree...")
    # Load data
    data, score = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    data_train, data_test, score_train, score_test = train_test_split(
        data, score, random_state=200)
    # Train model
    reg = RegressionTree()
    reg.fit(data=data_train, score=score_train, max_depth=5)
    # Show rules
    print(reg)
    # Model evaluation
    get_r2(reg, data_test, score_test)
```
## 3.2 效果展示
最终生成了15条规则，拟合优度0.776，运行时间634毫秒，效果还算不错~
![regression_tree](https://github.com/tushushu/imylu/blob/master/pic/regression_tree.png)

## 3.3 工具函数
本人自定义了一些工具函数，可以在github上查看  
[utils](https://github.com/tushushu/imylu/tree/master/imylu/utils)  

1. run_time - 测试函数运行时间  
2. load_boston_house_prices - 加载波士顿房价数据  
3. train_test_split - 拆分训练集、测试集  
4. get_r2 - 计算拟合优度  


# 总结
回归树的原理：  
损失最小化，平均值大法。
最佳行与列，效果顶呱呱。

回归树的实现：  
一顿操作猛如虎，加减乘除二叉树。

