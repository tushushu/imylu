
提到GBDT回归相信大家应该都不会觉得陌生（不陌生你点进来干嘛[捂脸]），本文就GBDT回归的基本原理进行讲解，并手把手、肩并肩地带您实现这一算法。

完整实现代码请参考本人的p...哦不是...github：
[gbdt_base.py](https://github.com/tushushu/imylu/blob/master/imylu/ensemble/gbdt_base.py)
[gbdt_regressor.py](https://github.com/tushushu/imylu/blob/master/imylu/ensemble/gbdt_regressor.py)
[gbdt_regressor_example.py](https://github.com/tushushu/imylu/blob/master/examples/gbdt_regressor_example.py)


# 1. 原理篇
我们用人话而不是大段的数学公式来讲讲GBDT回归是怎么一回事。

## 1.1 温故知新
回归树是GBDT的基础，之前的一篇文章曾经讲过回归树的原理和实现。链接如下：
[regression_tree.md](https://github.com/tushushu/imylu/blob/master/docs_cn/regression_tree.md)

## 1.2 预测年龄
仍然以预测同事年龄来举例，从《回归树》那篇文章中我们可以知道，如果需要通过一个常量来预测同事的年龄，平均值是最佳选择之一。

## 1.3 年龄的残差
我们不妨假设同事的年龄分别为5岁、6岁、7岁，那么同事的平均年龄就是6岁。所以我们用6岁这个常量来预测同事的年龄，即[6, 6, 6]。每个同事年龄的残差 = 年龄 - 预测值 = [5, 6, 7] - [6, 6, 6]，所以残差为[-1, 0, 1]

## 1.4 预测年龄的残差
为了让模型更加准确，其中一个思路是让残差变小。如何减少残差呢？我们不妨对残差建立一颗回归树，然后预测出准确的残差。假设这棵树预测的残差是[-0.9, 0, 0.9]，将上一轮的预测值和这一轮的预测值求和，每个同事的年龄 = [6, 6, 6] + [-0.9, 0, 0.9] = [5.1, 6, 6.9]，显然与真实值[5, 6, 7]更加接近了， 年龄的残差此时变为[-0.1, 0, 0.1]，预测的准确性得到了提升。

## 1.5 GBDT
重新整理一下思路，假设我们的预测一共迭代3轮
年龄：[5, 6, 7]

第1轮预测：[6, 6, 6] (平均值)

第1轮残差：[-1, 0, 1]

第2轮预测：[6, 6, 6] (平均值) + [-0.9, 0, 0.9] (第1颗回归树) = [5.1, 6, 6.9]

第2轮残差：[-0.1, 0, 0.1]

第3轮预测：[6, 6, 6] (平均值) + [-0.9, 0, 0.9] (第1颗回归树) + [-0.08, 0, 0.07] (第2颗回归树)  = [5.02, 6,  6.97]

第3轮残差：[-0.08, 0, 0.03]

看上去残差越来越小，而这种预测方式就是GBDT算法。

## 1.6 公式推导
看到这里，相信您对GBDT已经有了直观的认识。这么做有什么科学依据么，为什么残差可以越来越小呢？前方小段数学公式低能预警。

1. 假设要做m轮预测，预测函数为Fm，初始常量或每一轮的回归树为fm，输入变量为X，有：  
$F_m(X) = F_{m-1}(X) + f_m(X)$  

2. 设要预测的变量为y，采用MSE作为损失函数：  
$Loss(y, F_m(X)) = \Large\frac{1}{m}\normalsize\sum_{i=0}^m((y_i - F_m(x_i)) ^ 2)$

3. 我们知道泰勒公式的一阶展开式是长成这个样子滴：  
$f(x + x_0) = f(x) + f'(x) * x_0$

4. 如果：  
$f(x) = g'(x)$

5. 那么，根据式3和式4可以得出：  
$g'(x + x_0) = g'(x) + g''(x) * x_0$

6. 根据式2可以知道，损失函数的一阶偏导数为:  
$Loss'(y, F_m(X)) = -\Large\frac{2}{n}\normalsize\sum_{i=0}^m(y_i - F_m(x_i))$

7. 根据式6可以知道，损失函数的二阶偏导数为：  
$Loss''(y, F_m(X)) = 2$

8. 蓄力结束，开始放大招。根据式1，损失函数的一阶导数为：  
$Loss'(y, F_m(X)) = Loss'(y, F_{m-1}(X) + f_m(X))$

9. 根据式5，将式8进一步展开为：  
$Loss'(y, F_m(X))= Loss'(y, F_{m-1}(X)) + Loss''(y, F_{m-1}(X)) * f_m(X)$

10. 令式9，即损失函数的一阶导数为0，那么：  
$f_m(X) = - Loss'(y, F_{m-1}(X)) / Loss''(y, F_{m-1}(X))$

11. 将式6，式7代入式9得到：  
$f_m(X) = \Large\frac{1}{n}\normalsize\sum_{i=0}^m(y_i - F_{m-1}(x_i))$

因此，我们需要通过用第m-1轮残差的均值来得到函数fm，进而优化函数Fm。而回归树的原理就是通过最佳划分区域的均值来进行预测。所以fm可以选用回归树作为基础模型，将初始值，m-1颗回归树的预测值相加便可以预测y。

# 2. 实现篇
本人用全宇宙最简单的编程语言——Python实现了GBDT回归算法，没有依赖任何第三方库，便于学习和使用。简单说明一下实现过程，更详细的注释请参考本人github上的代码。
## 2.1 导入回归树类
回归树是我之前已经写好的一个类，在之前的文章详细介绍过，代码请参考：
[regression_tree.py](https://github.com/tushushu/imylu/blob/master/imylu/tree/regression_tree.py)
```Python
from ..tree.regression_tree import RegressionTree
```
## 2.2 创建GradientBoostingBase类
初始化，存储回归树、学习率、初始预测值和变换函数。（注：回归不需要做变换，因此函数的返回值等于参数）
```Python
class GradientBoostingBase(object):
    def __init__(self):
        self.trees = None
        self.lr = None
        self.init_val = None
        self.fn = lambda x: x
```

## 2.3 计算初始预测值
初始预测值即y的平均值。
```Python
def _get_init_val(self, y):
    return sum(y) / len(y)
```

## 2.4 计算残差
```Python
def _get_residuals(self, y, y_hat):
    return [yi - self.fn(y_hat_i) for yi, y_hat_i in zip(y, y_hat)]
```

## 2.5 训练模型
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

## 2.6 预测一个样本
```Python
def _predict(self, Xi):
    return self.fn(self.init_val + sum(self.lr * tree._predict(Xi) for tree in self.trees))
```

## 2.7 预测多个样本
```Python
def predict(self, X):
    return [self._predict(Xi) for Xi in X]
```

# 3 效果评估
## 3.1 main函数
使用著名的波士顿房价数据集，按照7:3的比例拆分为训练集和测试集，训练模型，并统计准确度。
```Python
@run_time
def main():
    print("Tesing the accuracy of GBDT regressor...")

    X, y = load_boston_house_prices()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10)

    reg = GradientBoostingRegressor()
    reg.fit(X=X_train, y=y_train, n_estimators=4,
            lr=0.5, max_depth=2, min_samples_split=2)

    get_r2(reg, X_test, y_test)
```
## 3.2 效果展示
最终拟合优度0.851，运行时间5.0秒，效果还算不错~
![gbdt_regressor.png](https://github.com/tushushu/imylu/blob/master/pic/gbdt_regressor.png)

## 3.3 工具函数
本人自定义了一些工具函数，可以在github上查看
[utils](https://github.com/tushushu/imylu/tree/master/imylu/utils)  
1. run_time - 测试函数运行时间  
2. load_boston_house_prices - 加载波士顿房价数据  
3. train_test_split - 拆分训练集、测试集  
4. get_r2 - 计算拟合优度


# 总结
GBDT回归的原理：平均值加回归树

GBDT回归的实现：加加减减for循环

