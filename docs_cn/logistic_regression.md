
提到逻辑回归相信大家应该都不会觉得陌生（不陌生你点进来干嘛[捂脸]），本文就逻辑回归的基本原理进行讲解，并手把手、肩并肩地带您实现这一算法。

完整实现代码请参考本人的p...哦不是...github：  
[regression_base.py](https://github.com/tushushu/imylu/blob/master/imylu/linear_model/regression_base.py)  
[logistic_regression.py](https://github.com/tushushu/imylu/blob/master/imylu/linear_model/logistic_regression.py)  
[logistic_regression_example.py](https://github.com/tushushu/imylu/blob/master/examples/logistic_regression_example.py)  


# 1. 原理篇
我们用人话而不是大段的数学公式来讲讲逻辑回归是怎么一回事。

## 1.1 梯度下降法
请参考我的另一篇文章，在这里就不赘述。链接如下：  
[gradient_decent.md](https://github.com/tushushu/pads/blob/master/docs_cn/gradient_decent.md)

## 1.2 线性回归
请参考我的另一篇文章，在这里就不赘述。链接如下：  
[linear_regression.md](https://github.com/tushushu/imylu/blob/master/docs_cn/linear_regression.md)

## 1.3 Sigmoid函数
Sigmoid函数的表达式是：  
$f(x) = \Large\frac{1}{1 + e^{-x}}$  

不难得出：  
$\lim\limits_{x\rightarrow{-\infty}}\Large\frac{1}{1 + e^{-x}}\normalsize = 0$  

$\lim\limits_{x\rightarrow{+\infty}}\Large\frac{1}{1 + e^{-x}}\normalsize = 1$  

$f'(x) = f(x) * (1 - f(x))$  

所以，Sigmoid函数的值域是(0, 1)，导数为y * (1 - y)

## 1.4 线性回归与逻辑回归
回归问题的值域是(-∞, +∞)，用线性回归可以进行预测。而分类问题的值域是[0, 1]，显然不可以直接用线性回归来预测这类问题。如果把线性回归的输出结果外面再套一层Sigmoid函数，正好可以让值域落在0和1之间，这样的算法就是逻辑回归。

## 1.5 最小二乘法
那么逻辑回归的损失函数是什么呢，根据之前线性回归的经验。  
用MSE作为损失函数，有  
$L = \large\frac{1}{m}\normalsize\sum_{1}^{m}(Y_{i} - \large\frac{1}{1 + e^{-WX_{i} - b}}\normalsize)^2$  
网上很多文章都说这个函数是非凸的，不可以用梯度下降来优化，为什么非凸也没见人给出个证明。我一开始是不信的，后来对损失函数求了二阶导之后...发现求导太麻烦了，所以我还是信了吧。

## 1.6 极大似然估计
既然不能用最小二乘法，那么肯定是有方法求解的，极大似然估计闪亮登场。前方小段数学公式低能预警：  
线性函数  
1. $z = WX + b$   
Sigmoid函数   
2. $\hat y = \large\frac{1}{1 + e^{-z}}$  
似然函数   
3. $P(Y | X, W, b) = \prod_{1}^{m} \hat y_{i}^{y_{i}} * (1-\hat y_{i})^{1-y_{i}}$  
对似然函数两边取对数的负值  
4. $L = -\sum_{1}^{m}(y_{i} * log \hat y_{i} + (1-y_{i}) * log(1-\hat y_{i}))$  
对1式求导   
5. $\large\frac{\mathrm{d}Z}{\mathrm{d}W}\normalsize=X$  
对2式求导  
6. $\large\frac{\mathrm{d}\hat{y}}{\mathrm{d}z}\normalsize=z * (1 - z)$  
对3式求导  
7. $\large\frac{\mathrm{d}L}{\mathrm{d}\hat{y}}\normalsize=y/\hat{y} - (1-y)/(1-\hat{y})$  
8. $\large\frac{\mathrm{d}Z}{\mathrm{d}b}\normalsize=1$  


根据5, 6, 7式:  
9. $\large\frac{\mathrm{d}L}{\mathrm{d}W}\normalsize=-\sum(y - \hat{y}) * X$ 

根据6, 7, 8式:  
10. $\large\frac{\mathrm{d}L}{\mathrm{d}W}\normalsize=-\sum(y - \hat{y})$ 



# 2. 实现篇
本人用全宇宙最简单的编程语言——Python实现了逻辑回归算法，没有依赖任何第三方库，便于学习和使用。简单说明一下实现过程，更详细的注释请参考本人github上的代码。

## 2.1 创建RegressionBase类
初始化，存储权重weights和偏置项bias。
```Python
class RegressionBase(object):
    def __init__(self):
        self.bias = None
        self.weights = None
```

## 2.2 创建LogisticRegression类
初始化，继承RegressionBase类。
```Python
class LogisticRegression(RegressionBase):
    def __init__(self):
        RegressionBase.__init__(self)
```

## 2.3 预测一个样本
```Python
def _predict(self, Xi):
    z = sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias
    return sigmoid(z)
```

## 2.4 计算梯度
根据损失函数的一阶导数计算梯度。
```Python
def _get_gradient_delta(self, Xi, yi):
    y_hat = self._predict(Xi)
    bias_grad_delta = yi - y_hat
    weights_grad_delta = [bias_grad_delta * Xij for Xij in Xi]
    return bias_grad_delta, weights_grad_delta
```

## 2.5 批量梯度下降
正态分布初始化weights，外层循环更新参数，内层循环计算梯度。
```Python
def _batch_gradient_descent(self, X, y, lr, epochs):
    m, n = len(X), len(X[0])
    self.bias = 0
    self.weights = [normalvariate(0, 0.01) for _ in range(n)]

    for _ in range(epochs):
        bias_grad = 0
        weights_grad = [0 for _ in range(n)]

        for i in range(m):
            bias_grad_delta, weights_grad_delta = self._get_gradient_delta(
                X[i], y[i])
            bias_grad += bias_grad_delta
            weights_grad = [w_grad + w_grad_d for w_grad, w_grad_d
                            in zip(weights_grad, weights_grad_delta)]

        self.bias += lr * bias_grad * 2 / m
        self.weights = [w + lr * w_grad * 2 / m for w,
                        w_grad in zip(self.weights, weights_grad)]
```

## 2.6 随机梯度下降
正态分布初始化weights，外层循环迭代epochs，内层循环随机抽样计算梯度。
```Python
def _stochastic_gradient_descent(self, X, y, lr, epochs, sample_rate):
    m, n = len(X), len(X[0])
    k = int(m * sample_rate)
    self.bias = 0
    self.weights = [normalvariate(0, 0.01) for _ in range(n)]

    for _ in range(epochs):
        for i in sample(range(m), k):
            bias_grad, weights_grad = self._get_gradient_delta(X[i], y[i])
            self.bias += lr * bias_grad
            self.weights = [w + lr * w_grad for w,
                            w_grad in zip(self.weights, weights_grad)]
```

## 2.7 训练模型
使用批量梯度下降或随机梯度下降训练模型。
```Python
def fit(self, X, y, lr, epochs, method="batch", sample_rate=1.0):
    assert method in ("batch", "stochastic")
    if method == "batch":
        self._batch_gradient_descent(X, y, lr, epochs)
    if method == "stochastic":
        self._stochastic_gradient_descent(X, y, lr, epochs, sample_rate)
```

## 2.8 预测多个样本
```Python
def predict(self, X, threshold=0.5):
    return [int(self._predict(Xi) >= threshold) for Xi in X]
```

# 3 效果评估
## 3.1 main函数
使用著名的乳腺癌数据集，按照7:3的比例拆分为训练集和测试集，训练模型，并统计准确度。
```Python
def main():
    @run_time
    def batch():
        print("Tesing the performance of LogisticRegression(batch)...")
        clf = LogisticRegression()
        clf.fit(X=X_train, y=y_train, lr=0.05, epochs=200)
        model_evaluation(clf, X_test, y_test)

    @run_time
    def stochastic():
        print("Tesing the performance of LogisticRegression(stochastic)...")
        clf = LogisticRegression()
        clf.fit(X=X_train, y=y_train, lr=0.01, epochs=200,
                method="stochastic", sample_rate=0.5)
        model_evaluation(clf, X_test, y_test)

    X, y = load_breast_cancer()
    X = min_max_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    batch()
    stochastic()
```

## 3.2 效果展示
批量梯度下降AUC 0.984，运行时间677.9 毫秒；
随机梯度下降AUC 0.997，运行时间437.6 毫秒。
效果还算不错~
![logistic_regression.png](https://github.com/tushushu/imylu/blob/master/pic/logistic_regression.png)

## 3.3 工具函数
本人自定义了一些工具函数，可以在github上查看  
[utils](https://github.com/tushushu/imylu/tree/master/imylu/utils)  

1. run_time - 测试函数运行时间  
2. load_breast_cancer - 加载乳腺癌数据  
3. train_test_split - 拆分训练集、测试集  
4. model_evaluation - 计算AUC，准确度，召回率
5. min_max_scale - 归一化


# 总结
逻辑回归的原理：线性回归结合Sigmoid函数
逻辑回归的实现：加减法，for循环。
