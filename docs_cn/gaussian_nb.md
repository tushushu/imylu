
好久不更新了，五一劳动节，劳动一下:)

提到朴素贝叶斯相信大家应该都不会觉得陌生（不陌生你点进来干嘛[捂脸]），本文就朴素贝叶斯的基本原理进行讲解，并手把手、肩并肩地带您实现这一算法。

完整实现代码请参考本人的p...哦不是...github：  
[gaussian_nb.py](https://github.com/tushushu/imylu/blob/master/imylu/probability_model/gaussian_nb.py)   
[gaussian_nb_example.py](https://github.com/tushushu/imylu/blob/master/examples/gaussian_nb_example.py)  


## 1. 原理篇
我们用人话而不是大段的数学公式来讲讲朴素贝叶斯是怎么一回事。

### 1.1 条件概率
条件概率故名思议就是在一定条件下发生某件事的概率。比如笔者向女生表白成功的概率是20%，记作P(A) = 20%，其中A代表“笔者向女生表白成功”。而笔者开着捷豹的前提下，向女生表白成功的概率是50%，则记作P(A|B) = 50%，其中B代表“笔者开着捷豹”。毕竟女生都喜欢小动物，像捷豹、路虎、宝马或者悍马什么的。咳咳，跑题了...这个P(A|B)就是条件概率了。

### 1.2 联合概率
那什么是联合概率呢，笔者开着捷豹且向女生表白成功的概率是1%，则记作P(AB) = 1%。您可能不禁要问，表白成功的概率不是20%吗？联合概率不是高达50%吗？为什么联合概率这么低？这个嘛，因为笔者特别穷，开捷豹的概率实在是太低了所以拖累了联合概率。

### 1.3 条件概率与联合概率的区别与联系
总结一下，条件概率就是在B的条件下，事件A发生的概率。而联合概率是事件A和B同时发生的概率。大家可以搞清楚区别了吧:)
而二者的联系，可以用如下公式表述：
P(AB) = P(A|B)P(B) = P(B|A)P(A)
即，联合概率 等于 条件概率 乘以 条件的概率。

### 1.4 全概率公式 
问题复杂一些，如果事件B不是一个条件，而是一堆条件，这些条件互斥且能穷尽所有可能。比如我白天向女生表白，黑夜向女生求爱（貌似有点押韵[奸笑]）。表白记作事件A，白天记作事件B1，黑夜记作事件B2。不难得到：  
P(A) = P(AB1) + P(AB2) = P(B1|A)P(A) + P(B2|A)P(A) = P(A|B1)P(B1) + P(A|B2)P(B2)  
其一般形式为：  
$P(A) = \normalsize\sum_{1}^{m}P(A|B_{i})P(B_{i})$

### 1.5 贝叶斯公式
既然算法的名字叫朴素贝叶斯，当当当当，我们的主角——贝叶斯公式隆重登场了。为了便于理解，我们不再使用事件A和事件B而是用机器学习常用的X, y来表示：    
$P(y_{i}|x) = P(x|y_{i})P(y_{i})/P(x)$
根据全概率公式，展开分母P(A)，得到：  
$P(y_{i}|x) = P(x|y_{i})P(y_{i})/\sum_{1}^{m}P(x|y_{j})P(y_{j})$

也就是说已知特征x，想要求解yi，只需要知道先验概率P(yi)，和似然度P(x|yi)，即可求解后验概率P(yi|x)。而对于同一个样本x，P(x)是一个常量，可以不参与计算。

### 1.6 高斯分布
如果x是连续变量，如何去估计似然度P(x|yi)呢？我们可以假设在yi的条件下，x服从高斯分布（正态分布）。根据正态分布的概率密度函数即可计算出P(x|yi)，公式如下：  
$P(x) = \large\frac{1}{\sigma\sqrt{2\pi}}\normalsize e^{-\frac{(x-\mu)^{2}}{2\sigma^2{}}}$

### 1.7 高斯朴素贝叶斯
如果x是多维的数据，那么我们可以假设P(x1|yi),P(x2|yi)...P(xn|yi)对应的事件是彼此独立的，这些值连乘在一起得到P(x|yi)，“彼此独立”也就是朴素贝叶斯的朴素之处。高斯朴素贝叶斯的原理大概就这么多了，是不是很简单？


## 2. 实现篇
本人用全宇宙最简单的编程语言——Python实现了朴素贝叶斯算法，便于学习和使用。简单说明一下实现过程，更详细的注释请参考本人github上的代码。

### 2.1 创建GaussianNB类
初始化，存储先验概率、训练集的均值、方差及label的类别数量。
```Python
class GaussianNB:
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None
```

### 2.2 计算先验概率
通过Python自带的Counter计算每个类别的占比，再将结果存储到numpy数组中。
```Python
def _get_prior(label: array)->array:
    cnt = Counter(label)
    prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
    return prior
```

### 2.3 计算训练集均值
每个label类别分别计算均值。
```Python
def _get_avgs(self, data: array, label: array)->array:
    return np.array([data[label == i].mean(axis=0) for i in range(self.n_class)])
```

### 2.4 计算训练集方差
每个label类别分别计算方差。
```Python
def _get_vars(self, data: array, label: array)->array:
    return np.array([data[label == i].var(axis=0) for i in range(self.n_class)])
```

### 2.5 计算似然度
通过高斯分布的概率密度函数计算出似然再连乘得到似然度。
```Python
def _get_likelihood(self, row: array)->array:
    return (1 / sqrt(2 * pi * self.vars) * exp(
        -(row - self.avgs)**2 / (2 * self.vars))).prod(axis=1)
```

### 2.6 训练模型
```Python
def fit(self, data: array, label: array):
    self.prior = self._get_prior(label)
    self.n_class = len(self.prior)
    self.avgs = self._get_avgs(data, label)
    self.vars = self._get_vars(data, label)
```

### 2.8 预测prob
用先验概率乘以似然度再归一化得到每个label的prob。
```Python
def predict_prob(self, data: array)->array:
    likelihood = np.apply_along_axis(self._get_likelihood, axis=1, arr=data)
    probs = self.prior * likelihood
    probs_sum = probs.sum(axis=1)
    return probs / probs_sum[:, None]
```

### 2.9 预测label
对于单个样本，取prob最大值所对应的类别，就是label的预测值。
```Python
def predict(self, data: array)->array:
    return self.predict_prob(data).argmax(axis=1)
```

## 3 效果评估
### 3.1 main函数
使用著名的乳腺癌数据集，按照7:3的比例拆分为训练集和测试集，训练模型，并统计准确度。
```Python
def main():
    print("Tesing the performance of Gaussian NaiveBayes...")
    data, label = load_breast_cancer()
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=100)
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    y_hat = clf.predict(data_test)
    acc = _get_acc(label_test, y_hat)
    print("Accuracy is %.3f" % acc)
```

### 3.2 效果展示
ACC 0.942，运行时间22 毫秒。
效果还算不错~
![gaussian_nb.png](https://github.com/tushushu/imylu/blob/master/pic/gaussian_nb.png)

### 3.3 工具函数
本人自定义了一些工具函数，可以在github上查看  
[utils](https://github.com/tushushu/imylu/tree/master/imylu/utils)  

1. run_time - 测试函数运行时间  
2. load_breast_cancer - 加载乳腺癌数据  
3. train_test_split - 拆分训练集、测试集  


## 总结
朴素贝叶斯的原理：贝叶斯公式
朴素贝叶斯的实现：加法、乘法、指数、开方。
