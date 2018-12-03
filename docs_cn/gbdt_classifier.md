
提到GBDT分类相信大家应该都不会觉得陌生（不陌生你点进来干嘛[捂脸]），本文就GBDT分类的基本原理进行讲解，并手把手、肩并肩地带您实现这一算法。

完整实现代码请参考本人的p...哦不是...github：
[gbdt_base.py](https://github.com/tushushu/imylu/blob/master/imylu/ensemble/gbdt_base.py)
[gbdt_classifier.py](https://github.com/tushushu/imylu/blob/master/imylu/ensemble/gbdt_classifier.py)
[gbdt_classifier_example.py](https://github.com/tushushu/imylu/blob/master/examples/gbdt_classifier_example.py)


# 1. 原理篇
我们用人话而不是大段的数学公式来讲讲GBDT分类是怎么一回事。

## 1.1 温故知新
GBDT分类只是在GBDT回归上做了一点点改造，而GBDT分类又是建立在回归树的基础上的。
之前写过一篇GBDT回归的文章，链接如下：  
[gbdt_regressor.md](https://github.com/tushushu/imylu/blob/master/docs_cn/gbdt_regressor.md)
之前写过一篇回归树的文章，链接如下：  
[regression_tree.md](https://github.com/tushushu/imylu/blob/master/docs_cn/regression_tree.md)

## 1.2 Sigmoid函数
如果对逻辑回归或者神经网络有所了解的话，那么对Sigmoid函数应该不会感到陌生，它的函数表达式是：  
$f(x) = \Large\frac{1}{1 + e^{-x}}$  

不难得出：  
$\lim\limits_{x\rightarrow{-\infty}}\Large\frac{1}{1 + e^{-x}}\normalsize = 0$  

$\lim\limits_{x\rightarrow{+\infty}}\Large\frac{1}{1 + e^{-x}}\normalsize = 1$  

$f'(x) = f(x) * (1 - f(x))$  

所以，Sigmoid函数的值域是(0, 1)，导数为y * (1 - y)

# 1.3 改造GBDT回归
根据《GBDT回归》可知，假设要做m轮预测，预测函数为Fm，初始常量或每一轮的回归树为fm，输入变量为X，有：  
$F_m(X) = F_{m-1}(X) + f_m(X)$  

由于是回归问题，函数F的值域在(-∞, +∞)，而二分类问题要求预测的函数值在(0, 1)，所以我们可以用Sigmoid函数将最终的预测值的值域控制在(0, 1)之间，其函数表达式如下：  
$p = \Large\frac{1}{1 + e^{-F_m(X)}}$

## 1.3 预测见面
以预测相亲是否见面来举例，见面用1表示，不见面用0表示。从《回归树》那篇文章中我们可以知道，如果需要通过一个常量来预测同事的年龄，平均值是最佳选择之一。那么预测二分类问题，这个常量该如何计算呢？我们简单证明一下：

1. z为我们要计算的常量：  
$z = F_m(X)$

2. GBDT分类器的函数表达式：  
$p = \Large\frac{1}{1 + e^{-z}}$

3. 二分类问题的似然函数：  
$Likelihood(p, y) = \prod_{i=0}^mp^{yi} * (1-p)^{1-yi}$  

4. 对式3两边求对数并乘以-1，得到损失函数：  
$Loss(p, y) = -\sum_{i=0}^{m}(yi * Logp + (1-yi) * Log(1-p))$

5. 对式4的p求导，得：  
$\Large\frac{\mathrm{d}L}{\mathrm{d}p}\normalsize = 
-\sum_{i=0}^{m}(yi/p - (1-yi)/(1-p))$

6. 对式2的z求导，得：  
$\Large\frac{\mathrm{d}p}{\mathrm{d}z}\normalsize= p * (1 - p)$

7. 根据式5和式6，得：  
$
\Large\frac{\mathrm{d}L}{\mathrm{d}z}\normalsize=
\Large\frac{\mathrm{d}L}{\mathrm{d}p} * 
\Large\frac{\mathrm{d}p}{\mathrm{d}z}\normalsize=
p * (1 - p) * -\sum_{i=0}^{m}(yi/p - (1-yi)/(1-p)
$  

8. 化简式7，得：  
$
\Large\frac{\mathrm{d}L}{\mathrm{d}z}\normalsize=
\sum_{i=0}^{m}p-\sum_{i=0}^{m}yi
$

9. 令式8等于0，最小化损失函数，那么：  
$p = \Large\frac{1}{m}\normalsize\sum_{i=0}^{m}yi$  

10. 将式2代入式9，得到：  
$z = log\Large\frac{\sum_{i=0}^{m}(yi)}{\sum_{i=0}^{m}(1-yi)}$

结论，如果要用一个常量来预测y，用log(sum(y)/sum(1-y))是一个最佳的选择。

## 1.4 见面的残差
我们不妨假设三个相亲对象是否见面分别为[1, 0, 1]，那么预测是否见面的初始值z = log((1+0+1)/(0+1+0)) = 0.693，所以我们用0.693这个常量来预测同事的年龄，即Sigmoid([0.693, 0.693, 0.693]) = [0.667, 0.667, 0.667]。每个相亲对象是否见面的残差 = 是否见面 - 预测值 = [1, 0, 1] - [0.667, 0.667, 0.667]，所以残差为[0.333, -0.667, 0.333]

## 1.5 预测见面的残差
为了让模型更加准确，其中一个思路是让残差变小。如何减少残差呢？我们不妨对残差建立一颗回归树，然后预测出准确的残差。假设这棵树预测的残差是[1, -0.5, 1]，将上一轮的预测值和这一轮的预测值求和，之后再求Sigmoid值，每个相亲对象是否见面 = Sigmoid([0.693, 0.693, 0.693] + [1, -0.5, 1]) = [0.845, 0.548, 0.845]，显然与真实值[1, 0, 1]更加接近了， 每个相亲对象是否见面的残差此时变为[0.155, -0.548, 0.155]，预测的准确性得到了提升。

## 1.6 GBDT
重新整理一下思路，假设我们的预测一共迭代3轮
是否见面：[1, 0, 1]

第1轮预测：Sigmoid([0.693, 0.693, 0.693] (初始值)) = [0.667, 0.667, 0.667]

第1轮残差：[0.333, -0.667, 0.333]

第2轮预测：Sigmoid([0.693, 0.693, 0.693] (初始值) + [1, -0.5, 1]) (第1颗回归树)) = Sigmoid([1.693, 0.193, 1.693]) = [0.845, 0.548, 0.845]

第2轮残差：[0.155, -0.548, 0.155]

第3轮预测：Sigmoid([0.693, 0.693, 0.693] (初始值) + [1, -0.5, 1] (第1颗回归树) + [2, -1, 2] (第2颗回归树))  = Sigmoid([3.693, -0.807,  3.693]) = [0.976, 0.309, 0.976]

第3轮残差：[0.024, -0.309, 0.024]

看上去残差越来越小，而这种预测方式就是GBDT算法。

## 1.7 公式推导
看到这里，相信您对GBDT已经有了直观的认识。这么做有什么科学依据么，为什么残差可以越来越小呢？前方小段数学公式低能预警。

1. 假设要做m轮预测，预测函数为Fm，初始常量或每一轮的回归树为fm，输入变量为X，有：  
$Z = F_m(X) = F_{m-1}(X) + f_m(X)$  
$P = 1 / (1 + e^{-Z})$

1. 设要预测的变量为y，采用极大似然函数的负对数作为损失函数：  
$Loss(y, F_m(X)) = \sum_{i=0}^{m}(y_i * logp_i + (1-y_i) * log(1-p_i))$

3. 我们知道泰勒公式的一阶展开式是长成这个样子滴：  
$f(x + x_0) = f(x) + f'(x) * x_0$

4. 如果：  
$f(x) = g'(x)$

5. 那么，根据式3和式4可以得出：  
$g'(x + x_0) = g'(x) + g''(x) * x_0$

6. 根据式2可以知道，损失函数的一阶偏导数为:  
$Loss'(y, F_m(X)) = \sum_{i=0}^{m}(y_i - p_i))$

1. 根据式6可以知道，损失函数的二阶偏导数为：  
$Loss''(y, F_m(X)) = \sum_{i=0}^{m}((p_i - 1) * p_i)$

8. 蓄力结束，开始放大招。根据式1，损失函数的一阶导数为：  
$Loss'(y, F_m(X)) = Loss'(y, F_{m-1}(X) + f_m(X))$

9. 根据式5，将式8进一步展开为：  
$Loss'(y, F_m(X))= Loss'(y, F_{m-1}(X)) + Loss''(y, F_{m-1}(X)) * f_m(X)$

10. 令式9，即损失函数的一阶导数为0，那么：  
$f_m(X) = - Loss'(y, F_{m-1}(X)) / Loss''(y, F_{m-1}(X))$

11. 将式6，式7代入式9得到：  
$f_m(X) = \Large\frac{\sum_{i=0}^{m}(y_i-p_i)}{\sum_{i=0}^{m}p_i * (1-p_i)}$

因此，我们需要通过用第m-1轮的预测值和残差来得到函数fm，进而优化函数Fm。而回归树的原理就是通过最佳划分区域的均值来进行预测，与GBDT回归不同，要把这个均值改为1.7式11。所以fm可以选用回归树作为基础模型，将初始值，m-1颗回归树的预测值相加再求Sigmoid值便可以预测y。

# 2. 实现篇
本人用全宇宙最简单的编程语言——Python实现了GBDT分类算法，没有依赖任何第三方库，便于学习和使用。简单说明一下实现过程，更详细的注释请参考本人github上的代码。
## 2.1 导入回归树类
回归树是我之前已经写好的一个类，在之前的文章详细介绍过，代码请参考：
[regression_tree.py](https://github.com/tushushu/imylu/blob/master/imylu/tree/regression_tree.py)
```Python
from ..tree.regression_tree import RegressionTree
```
## 2.2 创建GradientBoostingBase类
初始化，存储回归树、学习率、初始预测值和变换函数。
```Python
class GradientBoostingBase(object):
    def __init__(self):
        self.trees = None
        self.lr = None
        self.init_val = None
        self.fn = lambda x: sigmoid(x)
```

## 2.3 计算初始预测值
初始预测值，见1.7式10。
```Python
def _get_init_val(self, y):
    n = len(y)
    y_sum = sum(y)
    return log((y_sum) / (n - y_sum))
```

## 2.4 匹配叶结点
计算训练样本属于回归树的哪个叶子结点。
```Python
def _match_node(self, row, tree):
    nd = tree.root
    while nd.left and nd.right:
        if row[nd.feature] < nd.split:
            nd = nd.left
        else:
            nd = nd.right
    return nd
```

## 2.5 获取叶节点
获取一颗回归树的所有叶子结点。
```Python
def _get_leaves(self, tree):
    nodes = []
    que = [tree.root]
    while que:
        node = que.pop(0)
        if node.left is None or node.right is None:
            nodes.append(node)
            continue
        left_node = node.left
        right_node = node.right
        que.append(left_node)
        que.append(right_node)
    return nodes
```

## 2.6 划分区域
将回归树的叶子结点，其对应的所有训练样本存入字典。
```Python
def _divide_regions(self, tree, nodes, X):
    regions = {node: [] for node in nodes}
    for i, row in enumerate(X):
        node = self._match_node(row, tree)
        regions[node].append(i)
    return regions
```

## 2.7 计算预测值
见1.7式11。
```Python
def _get_score(self, idxs, y_hat, residuals):
    numerator = denominator = 0
    for idx in idxs:
        numerator += residuals[idx]
        denominator += y_hat[idx] * (1 - y_hat[idx])
    return numerator / denominator
```

## 2.8 更新预测值
更新回归树各个叶节点的预测值。
```Python
def _update_score(self, tree, X, y_hat, residuals):
    nodes = self._get_leaves(tree)
    regions = self._divide_regions(tree, nodes, X)
    for node, idxs in regions.items():
        node.score = self._get_score(idxs, y_hat, residuals)
    tree._get_rules()
```

## 2.9 计算残差
```Python
def _get_residuals(self, y, y_hat):
    return [yi - self.fn(y_hat_i) for yi, y_hat_i in zip(y, y_hat)]
```

## 2.10 训练模型
训练模型的时候需要注意以下几点：
1. 控制树的最大深度max_depth；
2. 控制分裂时最少的样本量min_samples_split；
3. 训练每一棵回归树的时候要乘以一个学习率lr，防止模型过拟合；
4. 对样本进行抽样的时候要采用有放回的抽样方式。
```Python
def fit(self, X, y, n_estimators, lr, max_depth, min_samples_split, subsample=None):
    self.init_val = self._get_init_val(y)

    n = len(y)
    y_hat = [self.init_val] * n
    residuals = self._get_residuals(y, y_hat)

    self.trees = []
    self.lr = lr
    for _ in range(n_estimators):
        idx = range(n)
        if subsample is not None:
            k = int(subsample * n)
            idx = choices(population=idx, k=k)
        X_sub = [X[i] for i in idx]
        residuals_sub = [residuals[i] for i in idx]
        y_hat_sub = [y_hat[i] for i in idx]

        tree = RegressionTree()
        tree.fit(X_sub, residuals_sub, max_depth, min_samples_split)

        self._update_score(tree, X_sub, y_hat_sub, residuals_sub)

        y_hat = [y_hat_i + lr * res_hat_i for y_hat_i,
                    res_hat_i in zip(y_hat, tree.predict(X))]

        residuals = self._get_residuals(y, y_hat)
        self.trees.append(tree)
```

## 2.11 预测一个样本
```Python
def _predict(self, Xi):
    return self.fn(self.init_val + sum(self.lr * tree._predict(Xi) for tree in self.trees))
```

## 2.12 预测多个样本
```Python
def predict(self, X):
    return [int(self._predict(Xi) >= threshold) for Xi in X]
```

# 3 效果评估
## 3.1 main函数
使用著名的乳腺癌数据集，按照7:3的比例拆分为训练集和测试集，训练模型，并统计准确度。
```Python
@run_time
def main():

    print("Tesing the accuracy of GBDT classifier...")

    X, y = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train, n_estimators=2,
            lr=0.8, max_depth=3, min_samples_split=2)

    get_acc(clf, X_test, y_test)
```
## 3.2 效果展示
最终准确度93.082%，运行时间14.9秒，效果还算不错~
![gbdt_classifier.png](https://github.com/tushushu/imylu/blob/master/pic/gbdt_classifier.png)

## 3.3 工具函数
本人自定义了一些工具函数，可以在github上查看：  
[utils](https://github.com/tushushu/imylu/tree/master/imylu/utils)  
1. run_time - 测试函数运行时间  
2. load_breast_cancer - 加载乳腺癌数据  
3. train_test_split - 拆分训练集、测试集  
4. get_acc - 计算准确度


# 总结
GBDT分类的原理：GBDT回归加Sigmoid

GBDT分类的实现：一言难尽[哭]

