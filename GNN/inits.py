import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init.统一初始化"""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    # random_uniform():返回shape*shape的矩阵，产生于minval和maxval之间，产生的值是均匀分布(随机均匀分布)的。
    # 此句：定义tensor类对象initial
    return tf.Variable(initial, name=name)
    # 此句：返回tensor类对象

    # tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
    # tf.Variable():
    #   通过创建Variable类的实例向graph中添加变量。Variable()需要初始值，一旦初始值确定，那么该变量的类型和形状都确定了。
    #   更改值通过assign方法。想要改变形状，可以使用一个assign op，令validate_shape=False。
    #   通过Variable()生成的variables就是一个tensor，可以作为graph中其他op的输入。
    #   另外，Tensor类重载的所有操作符都被转载到此variables中，所以可以通过对变量调用方法，将节点添加到图形中。


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init.权重初始化(Xavier初始化)"""
    # glorot认为优秀的初始化：各个层的激活值h（输出值）的方差要保持一致、各个层对状态Z的梯度的方差要保持一致。
    # (z为激活函数的输入，在网络层内部。)
    # Xavier初始化的基本思想是：若对于一层网络的输出和输出可以保持正态分布且方差相近，
    #                          这样就可以避免输出趋向于0，从而避免梯度弥散情况。

    init_range = np.sqrt(6.0/(shape[0]+shape[1]))    # 定义n维数组类(ndarray)对象init_range,接收np.sqrt()返回的平方根。
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    # 定义tensor类对象initial，大小为shape*shape，其中的元素在(-init_range)到init_range之间，元素数据为浮点型
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros.生成0矩阵(张量(tensor))"""
    # 可能是对偏置项的初始化
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones.生成元素全为1的矩阵(张量(tensor))"""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
