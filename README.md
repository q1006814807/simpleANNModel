# simpleANNModel
a simple and general artificial neural network model

你好！感谢你的到访，这是我写的一个基于mnist数据集的简单的神经网络框架，本人代码见拙，还请多多指教！

1）只支持普通的全连接层以及softmax分类输出

2）loss为平方损失

3）示例参数为：
  14*14 + 64 + 10
  linear + tanh + softmax

  [training]accuracy     = 0.98126
  [training]accum loss   = 163.727/10000张
  [validation]accuracy   = 0.9636
  [validation]accum loss = 280.659/10000张
  [test]accuracy         = 0.9601
  [test]accum loss       = 305.825/10000张
