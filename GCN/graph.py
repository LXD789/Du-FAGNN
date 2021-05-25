# 如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，
# 也就是说它在当前版本中不是语言标准，那么我们如果想要使用的话就要从__future__模块导入
from __future__ import print_function  # print()函数

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import h5py


# 定义基本的图卷积类
# Keras自定义层要实现build方法、call方法和compute_output_shape(input_shape)方法
class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    # 构造函数
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):   # kernel_initializer='glorot_uniform',bias_initializer='zeros',
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            # pop()函数用于删除列表中某元素，并返回该元素的值
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # 施加在权重上的正则项
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # 施加在偏置向量上的正则项
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # 施加在输出上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # 对主权重矩阵进行约束
        self.kernel_constraint = constraints.get(kernel_constraint)
        # 对偏置向量进行约束
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        # 下边三个属性是我自己后期因为有Instance attribute kernel defined outside __init__警告而加上的
        #                   ————2021.3.16
        self.kernel = None
        self.bias = None
        self.built = None

        self.support = support
        assert support >= 1

    # 计算输出的形状
    # 如果自定义层更改了输入张量的形状，则应该在这里定义形状变化的逻辑
    # 让Keras能够自动推断各层的形状
    def compute_output_shape(self, input_shapes):
        # 特征矩阵形状
        features_shape = input_shapes[0]
        # 输出形状为(批大小, 输出维度)
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    # 定义层中的参数
    def build(self, input_shapes):
        # 特征矩阵形状
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        # 特征维度
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # 如果存在偏置
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # 必须设定self.bulit = True
        self.built = True

    # 编写层的功能逻辑
    def call(self, inputs, mask=None):
        features = inputs[0]  # 特征
        basis = inputs[1:]  # 对称归一化的邻接矩阵

        # 多个图的情况
        supports = list()
        for i in range(self.support):
            # A * X
            supports.append(K.dot(basis[i], features))
        # 将多个图的结果按行拼接
        supports = K.concatenate(supports, axis=1)
        # A * X * W
        output = K.dot(supports, self.kernel)

        if self.bias:
            # A * X * W + b
            output += self.bias
        return self.activation(output)  # 此处的activation为relu(此处的relu不是最终的)

    # 定义当前层的配置信息
    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphConvolution1(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    # 构造函数
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            # pop()函数用于删除列表中某元素，并返回该元素的值
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution1, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # 施加在权重上的正则项
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # 施加在偏置向量上的正则项
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # 施加在输出上的正则项
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # 对主权重矩阵进行约束
        self.kernel_constraint = constraints.get(kernel_constraint)
        # 对偏置向量进行约束
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        # 下边三个属性是我自己后期因为有Instance attribute kernel defined outside __init__警告而加上的
        #                   ————2021.3.16
        self.kernel = None
        self.bias = None
        self.built = None

        self.support = support
        assert support >= 1

    # 计算输出的形状
    # 如果自定义层更改了输入张量的形状，则应该在这里定义形状变化的逻辑
    # 让Keras能够自动推断各层的形状
    def compute_output_shape(self, input_shapes):
        # 特征矩阵形状
        features_shape = input_shapes[0]
        # 输出形状为(批大小, 输出维度)
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    # 定义层中的参数
    def build(self, input_shapes):
        # 特征矩阵形状
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        # 特征维度
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # 如果存在偏置
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # 必须设定self.bulit = True
        self.built = True

    # 编写层的功能逻辑
    def call(self, inputs, mask=None):
        features = inputs[0]  # 特征
        basis = inputs[1:]  # 对称归一化的邻接矩阵

        # 多个图的情况
        supports = list()
        for i in range(self.support):
            # A * X
            supports.append(K.dot(basis[i], features))
        # 将多个图的结果按行拼接
        supports = K.concatenate(supports, axis=1)
        # A * X * W
        output = K.dot(supports, self.kernel)

        if self.bias:
            # A * X * W + b
            output += self.bias
        return self.activation(output)  # 此处的activation为relu(此处的relu不是最终的)

    # 定义当前层的配置信息
    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }

        base_config = super(GraphConvolution1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

