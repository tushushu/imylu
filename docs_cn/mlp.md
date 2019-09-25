
提到全连接神经网络相信大家应该都不会觉得陌生（不陌生你点进来干嘛[捂脸]），本文就全连接神经网络的基本原理进行讲解，并手把手、肩并肩地带您实现这一算法。

完整实现代码请参考本人的p...哦不是...github：  
[neural_network](https://github.com/tushushu/imylu/tree/master/imylu/neural_network)    


# 1. 原理篇
我们用人话而不是大段的数学公式来讲讲全连接神经网络是怎么一回事。

## 1.1 网络结构

## 1.2 Simoid函数

## 1.3 链式求导

## 1.4 向前传播

## 1.5 反向传播

## 1.6 算法用到的导数

## 1.7 拓扑排序



# 2. 实现篇
本人用全宇宙最简单的编程语言——Python实现了全连接神经网络算法，便于学习和使用。简单说明一下实现过程，更详细的注释请参考本人github上的代码。

## 2.1 创建BaseNode抽象类
作为各种类型Node的父类。
```Python
class BaseNode(ABC):
    def __init__(self, *inbound_nodes, name=None):
        self.name = name
        self._value = None
        self.inbound_nodes = [x for x in inbound_nodes]  # type: List[BaseNode]
        self.outbound_nodes = []  # type: List[BaseNode]
        self.gradients = dict()  # type: Dict[BaseNode, float]
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

    def __str__(self):
        size = str(self.value.shape) if self.value is not None else "null"
        return "<Node name: %s, Node size: %s>" % (self.name, size)

    @property
    def value(self)->ndarray:
        return self._value

    @value.setter
    def value(self, value):
        err_msg = "'value' has to be a number or a numpy array!"
        assert isinstance(value, (ndarray, int, float)), err_msg
        self._value = value

    @abstractmethod
    def forward(self):
        return

    @abstractmethod
    def backward(self):
        return
```

## 2.2 创建InputNode类
初始化，继承BaseNode类。
```Python
class InputNode(BaseNode):
    def __init__(self, value: ndarray, name=None):
        BaseNode.__init__(self, name=name)
        self.value = value
        self.indexes = None

    @property
    def value(self):
        err_msg = "Indexes is None!"
        assert self.indexes is not None, err_msg
        return self._value[self.indexes]

    @value.setter
    def value(self, value: ndarray):
        BaseNode.value.fset(self, value)

    def forward(self):
        return

    def backward(self):
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            self.gradients[self] += node.gradients[self]
```

## 2.3 创建LinearNode类
初始化，继承BaseNode类。
```Python
class LinearNode(BaseNode):
    def __init__(self, data: BaseNode, weights: WeightNode, bias: WeightNode, name=None):
        BaseNode.__init__(self, data, weights, bias, name=name)

    def forward(self):
        data, weights, bias = self.inbound_nodes
        self.value = np.dot(data.value, weights.value) + bias.value

    def backward(self):
        data, weights, bias = self.inbound_nodes
        self.gradients = {node: np.zeros_like(node.value) for node in self.inbound_nodes}
        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            self.gradients[data] += np.dot(grad_cost, weights.value.T)
            self.gradients[weights] += np.dot(data.value.T, grad_cost)
            self.gradients[bias] += np.sum(grad_cost, axis=0, keepdims=False)
```

## 2.4 创建MseNode类
初始化，继承BaseNode类。
```Python
class MseNode(BaseNode):
    def __init__(self, label: InputNode, pred: LinearNode, name=None):
        BaseNode.__init__(self, label, pred, name=name)
        self.n_label = None
        self.diff = None

    def forward(self):
        label, pred = self.inbound_nodes
        self.n_label = label.value.shape[0]
        self.diff = (label.value - pred.value).reshape(-1, 1)
        self.value = np.mean(self.diff**2)

    def backward(self):
        label, pred = self.inbound_nodes
        self.gradients[label] = (2 / self.n_label) * self.diff
        self.gradients[pred] = -self.gradients[label]
```

## 2.5 创建SigmoidNode类
初始化，继承BaseNode类。
```Python
class SigmoidNode(BaseNode):
    def __init__(self, input_node: LinearNode, name=None):
        BaseNode.__init__(self, input_node, name=name)

    @staticmethod
    def _sigmoid(arr: ndarray) -> ndarray:
        return 1. / (1. + np.exp(-arr))

    @staticmethod
    def _derivative(arr: ndarray) -> ndarray:
        return arr * (1 - arr)

    def forward(self):
        input_node = self.inbound_nodes[0]
        self.value = self._sigmoid(input_node.value)

    def backward(self):
        input_node = self.inbound_nodes[0]
        self.gradients = {input_node: np.zeros_like(input_node.value)}
        for output_node in self.outbound_nodes:
            grad_cost = output_node.gradients[self]
            self.gradients[input_node] += self._derivative(self.value) * grad_cost
```

## 2.6 创建WeightNode类
初始化，继承BaseNode类。
```Python
class WeightNode(BaseNode):
    def __init__(self, shape: Union[Tuple[int, int], int], name=None, learning_rate=None):
        BaseNode.__init__(self, name=name)
        if isinstance(shape, int):
            self.value = np.zeros(shape)
        if isinstance(shape, tuple):
            self.value = np.random.randn(*shape)
        self.learning_rate = learning_rate

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for node in self.outbound_nodes:
            self.gradients[self] += node.gradients[self]
        partial = self.gradients[self]
        self.value -= partial * self.learning_rate
```

## 2.7 

```Python
class MLP:
    def __init__(self):
        self.nodes_sorted = []
        self._learning_rate = None
        self.data = None
        self.prediction = None
        self.label = None
```

## 2.8 

```Python
def __str__(self):
    if not self.nodes_sorted:
        return "Network has not be trained yet!"
    print("Network informantion:\n")
    ret = ["learning rate:", str(self._learning_rate), "\n"]
    for node in self.nodes_sorted:
        ret.append(node.name)
        ret.append(str(node.value.shape))
        ret.append("\n")
    return " ".join(ret)
```

## 2.9 

```Python
@property
def learning_rate(self) -> float:
    return self._learning_rate

@learning_rate.setter
def learning_rate(self, learning_rate):
    self._learning_rate = learning_rate
    for node in self.nodes_sorted:
        if isinstance(node, WeightNode):
            node.learning_rate = learning_rate
```

## 2.10 

```Python
def topological_sort(self, input_nodes):
    nodes_sorted = []
    que = copy(input_nodes)
    unique = set()
    while que:
        node = que.pop(0)
        nodes_sorted.append(node)
        unique.add(node)
        for outbound_node in node.outbound_nodes:
            if all(x in unique for x in outbound_node.inbound_nodes):
                que.append(outbound_node)
    self.nodes_sorted = nodes_sorted
```

## 2.11 

```Python
def forward(self):
    assert self.nodes_sorted is not None, "nodes_sorted is empty!"
    for node in self.nodes_sorted:
        node.forward()

def backward(self):
    assert self.nodes_sorted is not None, "nodes_sorted is empty!"
    for node in self.nodes_sorted[::-1]:
        node.backward()

def forward_and_backward(self):
    self.forward()
    self.backward()
```

## 2.12 

```Python
def build_network(self, data: ndarray, label: ndarray, n_hidden: int, n_feature: int):
    weight_node1 = WeightNode(shape=(n_feature, n_hidden), name="W1")
    bias_node1 = WeightNode(shape=n_hidden, name="b1")
    weight_node2 = WeightNode(shape=(n_hidden, 1), name="W2")
    bias_node2 = WeightNode(shape=1, name="b2")
    self.data = InputNode(data, name="X")
    self.label = InputNode(label, name="y")
    linear_node1 = LinearNode(
        self.data, weight_node1, bias_node1, name="l1")
    sigmoid_node1 = SigmoidNode(linear_node1, name="s1")
    self.prediction = LinearNode(
        sigmoid_node1, weight_node2, bias_node2, name="prediction")
    MseNode(self.label, self.prediction, name="mse")
    input_nodes = [weight_node1, bias_node1,
                    weight_node2, bias_node2, self.data, self.label]
    self.topological_sort(input_nodes)
```

## 2.13 

```Python
def train_network(self, epochs: int, n_sample: int, batch_size: int):
    steps_per_epoch = n_sample // batch_size
    for i in range(epochs):
        loss = 0
        for _ in range(steps_per_epoch):
            indexes = choice(n_sample, batch_size, replace=True)
            self.data.indexes = indexes
            self.label.indexes = indexes
            self.forward_and_backward()
            loss += self.nodes_sorted[-1].value
        print("Epoch: {}, Loss: {:.3f}".format(
            i + 1, loss / steps_per_epoch))
    print()
```

## 2.14 

```Python
def pop_unused_nodes(self):
    for _ in range(len(self.nodes_sorted)):
        node = self.nodes_sorted.pop(0)
        if node.name in ("mse", "y"):
            continue
        self.nodes_sorted.append(node)
```

## 2.15 训练模型
使用批量梯度下降或随机梯度下降训练模型。
```Python
def fit(self, data: ndarray, label: ndarray, n_hidden: int, epochs: int,
        batch_size: int, learning_rate: float):
    label = label.reshape(-1, 1)
    n_sample, n_feature = data.shape
    self.build_network(data, label, n_hidden, n_feature)
    self.learning_rate = learning_rate
    print("Total number of samples = {}".format(n_sample))
    self.train_network(epochs, n_sample, batch_size)
    self.pop_unused_nodes()
```

## 2.16 预测多个样本
```Python
def predict(self, data: ndarray) -> ndarray:
    self.data.value = data
    self.data.indexes = range(data.shape[0])
    self.forward()
    return self.prediction.value.flatten()
```

# 3 效果评估
## 3.1 main函数
使用著名的波士顿房价数据集，按照7:3的比例拆分为训练集和测试集，训练模型，并统计准确度。
```Python
@run_time
def main():
    print("Tesing the performance of MLP....")
    data, label = load_boston_house_prices()
    data = min_max_scale(data)
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, random_state=20)
    reg = MLP()
    reg.fit(data=data_train, label=label_train, n_hidden=8,
            epochs=1000, batch_size=8, learning_rate=0.0008)
    get_r2(reg, data_test, label_test)
    print(reg)
```

## 3.2 效果展示
随机梯度下降拟合优度0.991，运行时间258.5毫秒。
效果还算不错~
![linear_regression.png](https://github.com/tushushu/imylu/blob/0.2/pic/linear_regression.png)

## 3.3 工具函数
本人自定义了一些工具函数，可以在github上查看  
[utils](https://github.com/tushushu/imylu/tree/0.2/imylu/utils)  

1. run_time - 测试函数运行时间  
2. load_boston_house_prices - 加载波士顿房价数据  
3. train_test_split - 拆分训练集、测试集  
4. get_r2 - 计算拟合优度 
5. array2str - 数组转字符串
6. min_max_scale - 归一化


# 总结
全连接神经网络的原理：
全连接神经网络的实现：
