## 深度学习入门：基于Python的理论与实现

这里罗列本书学习后，应该掌握的内容

#### 第三章   神经网络

* 感知网络的原型

  线性：与门、与非门、或门；非线性：异或门(多层感知机)；感知机参数：权重和偏置

* 阶跃函数、sigmoid函数的代码实现，两者区别，深度学习为何使用后者

  sigmoid函数平滑，在极大和极小值附近梯度接近0，但是阶跃函数的梯度除0外，都为0，神经网络学习无法进行

* 为什么神经网络使用非线性函数

  线性叠加，加深层无意义

* Relu函数

  h(x) = max(0, x)，比sigmoid的优势

* 权重符号格式

  右上角：表示第几层的权重、右下角：前面表示当前层的第几个神经元，后面表示前一层的第几个神经元

* 简单三层前向神经网络

  init_network、forward

* 回归用恒等函数，分类用Softmax函数，为什么?

  回归输出连续值，分类输出是离散值，同时考虑反向传播的局部梯度函数

* Softmax函数原理及代码实现，训练时使用，预测一般不使用

  防止溢出，每个值减去输入值的最大值；输出在0.0到1.0之间的实数，sum=1，相当于概率；利用了指数函数exp的单调递增性

* 正则化、批处理的minist简单网络

  像素值从0-255正则化到0.0-1.0；以0为中心分布；数据白化；批处理可以高速运算；get_data、init_network、predict、accuracy

#### 第四章   神经网络的学习

* 均分误差及其mini batch实现

* one-hot标签

  非one-hot标签类型时，y[np.arange(batch_size), t]

* cross-entropy error及其mini batch实现

  维度为1时，扩展维度

* 为什么使用损失函数而不用精度，类似激活函数不选择阶跃函数

  参数变化，精度可能不变或者不连续，梯度为0

* 数值微分实现及作用

  梯度校验

* 梯度下降的代码实现

  微分实现、学习率、迭代步数、更新参数

* mini batch的两层网络训练

  mini batch读取数据、predict、loss、accuracy、numerical_gradient、梯度更新、loss绘图

#### 第五章   误差反向传播法

* 误差反向传播算法的数学式和计算图理解

  数学式通过严格的数据证明，计算图将前向计算转换为变量+操作符的组合，拆解后逐级传递。比如“支付金额
  关于苹果的价格的导数”的值是2.2。这意味着，如果苹果的价格上涨1日元，最终的支付金额会增加2.2日元

* 链式求导，链式法则在反向传播中的使用

  输出关于输入的函数是多级的、复杂的，利用链式求导法则进行逐级计算

* 加法层和乘法层的前向和反向实现

  加法的反向传播只是将上游的值传给下游，并不需要正向传播的输入信号；乘法的反向传播会乘以输入信号的翻转值，所以要保存正向传播的输入信号

* 激活函数层的计算图及代码实现

  ReLU：如果正向传播时的输入值小于等于0，则反向传播的值为0，否则为dout * 1

  Sigmoid：y = 1 / (1+exp(-x))，dx = dout * (y * (1-y))

* 多维数组的Affine层，计算图及代码实现

  计算图： X:(N, 2)   W:(2, 3)    b:(3,)   Y:(N, 3)

  out = np.dot(X, W) + b

  db = np.sum(dout, axis=0)

  dW = np.dot(X.T, dout)

  dX = np.dot(dout, W.T)

* softmax with loss层实现

  详细的计算图比较复杂，可以记住简易版，反向传播结果为:（y1 −t1, y2 − t2, y3 − t3），即输出与标签差分

* 为什么分类使用交叉熵误差、回归使用平方和误差，反向传播的结果

  对应Softmax和恒等函数，都是为了获得到（y1 −t1, y2 − t2, y3 − t3）这种简单直接的误差

* 误差反向传播的实现

  loss = cross_entropy_error(y, t)

  dx = (y - t) / batch_size

* 数值微分和解析性数学求解对比及梯度确认

  数值微分简单，但是计算耗时，解析性数学求解高校利用误差反向传播法计算梯度高效，但参数量大、实现复杂，可以使用数值微分进行梯度一致性比较，即梯度确认

#### 第六章   与学习相关的技巧

* SGD实现及缺点

  stochastic gradient descent：沿梯度方向更新参数，并不断重复来靠近最优参数

  W $ \leftarrow $ W - lr * dW, learning rate选择0.001，0.003，0.01，0.03等

  简单、易实现、当函数的形状非均向时，搜索效率很低，可所有维度归一化

* momentum实现

  W $ \leftarrow $ W + v，v $ \leftarrow $ $ \alpha$v - $ \eta $dW, $ \eta $为学习率， momentum动量减弱了梯度方向的量能

* AdaGrad实现

  动态调整学习率，是学习率衰减的方法

  W $ \leftarrow $ W - $ \eta $  $ \frac {1} {\sqrt {h}}$ dW，h $ \leftarrow $ h + dW$ \bigodot $dW

  记录历史梯度平方和，学习越深入，更新幅度越小，最后会变为０，使用RMSPProp遗忘过去，反映当前

* Adam的原理

  将momentum和AdaGrad结合，三个参数：学习率、一次momentum系数(0.9)和二次momentum系数(0.999)

* 权重初始值不可以是0或者相同的值，为什么？

  0无法学习，相同值无法防止权重均一化、瓦解权重对称结构

* 梯度消失原理及解决方法，表现力受限

  激活函数等非线性函数，反向传播中梯度的值不断变小，最后消失＝０，各层的激活值的分布需要有广度，否则有所偏向的数据会出现梯度消失或者表现力受限问题

* sigmoid适合Xavier初始值，为什么

  前一层节点数n，则Xavier初始值为$\frac1{\sqrt n}$，tanh函数更好，因为关于原点对称，不对称的sigmoid输出总是正数，优化路径容易出现zigzag现象

* RELU函数适合He初始值，为什么

  Xavier初始值以激活函数中间部分的近似线性函数推导的，但Relu的负值区域值为0，为了更有广度，需要2倍系数，故采用He初始值$\frac2{\sqrt{n}}$。Xavier初始值时，随着层的加深，激活函数的输出偏向会变大，进而出现梯度消失问题，而He初始值时，各层分布广度相同，所以Relu使用更多

* batch norm优点、原理及实现

  通过调整各层的激活值分布，使其拥有适当的广度，对输入数据进行均值为0、方差为1（合适的分布）的
  正规化

  增大lr，快速学习；不依赖初始值；抑制过拟合

  通过计算图进行反向传播推导

* 过拟合是什么，产生原因及原理

  只拟合训练数据，无法泛化到测试数据

  大量参数或者训练数据少

* 权重衰减L1和L2抑制过拟合原理

  L1是绝对值之和，L2是平方范数，对大的权重进行惩罚

* dropout比上面L2优势，实现及原理

  网络模型复杂时使用，学习时随机删除神经元，一般比例0.5

* 超参包含哪些？验证方法

  神经元数量、batch大小、学习率、权重衰减

  分成训练数据、验证数据、测试数据

  大致选定一个范围，随机选择一个超参，然后不断进行精度评估，选出最有超参

#### 第七章   卷积神经网络

* CNN常见结构

  从Affine-ReLU结构变成了Convolution-ReLU-Pooling结果，一般CNN最后几层依然使用Affine-ReLU结果

* 全连接层的问题及卷积层加入

  全连接层将图像3维拉平成1维，忽略了图像的高、长、通道的3维形状

  CNN中，卷积层的输入输出称为特征图feature map

* 卷积运算、padding、stride概念及计算

  滤波器运算、padding调整输出大小、stride是滤波器使用的位置间隔

  OH = (H + 2P - FH) / S + 1，OW = (W + 2P - FW) / S + 1

* 多维卷积运算的处理流图

  滤波器的通道数只能设定为和输入数据的通道数相同的值

  (N, C, H, W) $\bigotimes$ (FN, C, FH, FW) $\rightarrow$ (N, FN, OH, OW) + (FN , 1, 1) $\rightarrow$ (N, FN, OH, OW)

  画出图像方块表示

* 池化max average，池化层特征

  Max池化，Average池化，无参数训练，通道数不变，对微小变化有鲁棒性

  OH = (H - PH) / S + 1，OW = (W - PW) / S + 1

* 卷积层和池化层的实现

  im2col将图像输入数据展开以适合滤波器，(N\*out_h\*out_w, -1)

  卷积层im2col、W.reshape(FN, -1).T 、np.dot

  池化层np.max(col, axis=1)

* CNN实现

  Conv-ReLU-Pooling-Affine-ReLU-Affine-Softmax

* CNN的可视化

  每一层的权重映射到0~255的图像值，第1层提取边缘和斑块，接下来纹理，后面更复杂的物体部件

* 各种CNN的发展变化

  LeNet：1998年，sigmoid、subsampling降采样

  AlexNet：2012年，ReLU、Pooling、LRN、Dropout、GPU普及

#### 第八章   深度学习

* 数据增强的方法

  旋转、垂直水平方向微小移动、crop、flip、亮度、放大缩小

* minist最好成绩网络不是很深，简单任务

* 加深网络层的必要性

  减少网络参数、扩大感受野、增加ReLU等激活函数的非线性表现力、分层传递信息、学习更加高效

* AlexNet、VGG、GoogleNet、ResNet各自特点

  AlexNet：2012年，ReLU、Pooling、LRN、Dropout、GPU普及

  VGG：2014年，有权重的层(Conv和Affine层)叠加到16或者19层、3*3小滤波器、简单且应用性强

  GoogleNet：2014年，不仅纵向有深度，横向也有宽度，Inception结构使用多个大小不同的滤波器

  ResNet：2015年，快捷结构横跨输入数据卷积层，将输入x合计到输出，F(x) = F(x) + x，反向传播信号无衰减传递

* 神经网络各层占用时间

  AlexNet中Conv层计算占GPU 95%，分布式学习、运算精度从64和32减为16

* R-CNN Selective Search，Faster RCNN，FCN，NIC（CNN+RNN），RNN，多模态处理

  RCNN使用Selective Search进行候选区域提取

  Faster RCNN使用CNN进行候选区域提取

  FCN(Fully Convolutional Network)用于语义分割，通过一次forward处理，对所有像素进行分类，全部Conv层，无Affine层，在最后的层基于双线性插值法扩大空间大小，即通过去卷积/逆卷积来实现

  NIC(Neural Image Caption)用于图像标题生成，CNN提取图像特征，RNN以特征为初始值递归生成文本

  多模态处理：组合图像和自然语言等多种信息进行的处理

* 图像生成GAN，DCGAN

  GAN(Generative Adversarial Network)：Generator和Discriminator两个网络竞争学习，无监督学习

* 无人驾驶SegNet

* 强化学习DQN

  代理（Agent）根据环境选择行动，然后通过这个行动改变环境。根据环境的变化，代理获得某种报酬。强化学习的目的是决定代理的行动方针，以获得更好的报酬