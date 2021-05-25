from layers import *
from metrics import *

# 以下两句意为：取出从终端传入的命令行输入参数
flags = tf.app.flags
# tf.app.flags主要用于处理命令行参数的解析工作，
#   其实可以理解为一个封装好了的argparse包（argparse是一种结构化的数据存储格式，类似于Json、XML）。
FLAGS = flags.FLAGS
# tf.app.flags.FLAGS可以从对应的命令行参数取出参数


class Model(object):      # 基础模型类——————————————————————————————————————
    def __init__(self, **kwargs):
        # **的作用：在函数定义中，收集关键字参数到一个新的字典，并将整个字典赋值给变量kwargs
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
            # assert（断言）用于判断一个表达式，在表达式条件为false的时候触发异常。
            # 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况。
            # 此句意为：若kwarg不在allowed_kwargs中，则触发异常，并打印出“'Invalid keyword argument: ' + 当前的kwarg”
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.embeddings = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):    # 在方法或属性的开头加下划线表示此方法为私有方法，无法在类外访问。
        raise NotImplementedError
    # NotImplementedError: 子类没有实现父类要求一定要实现的方法时会抛出的异常。
    # raise：为了引发异常，可以使用一个类（应该是Exception的子类）或者实例参数调用raise语句。
    #        raise可以实现报出错误的功能，报错的条件可以由程序员自己去定制。

    # 在面向对象编程中，可以先预留一个方法接口不实现，在其子类中实现。如果要求其子类一定要实现，
    # 不实现的时候会导致问题，那么采用raise的方式就很好。而此时产生的问题分类是NotImplementedError。

    def build(self):
        """ Wrapper for _build().   _build()的装饰器
        装饰器：其作用就是为已经存在的函数或对象添加额外的功能。装饰器本质上是一个Python函数，
            它可以让其他函数在不需要做任何代码变动的前提下增加额外功能，装饰器的返回值也是一个函数对象。"""
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model建立顺序层模型
        self.activations = [self.inputs]
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.embeddings = self.activations[-2]
        self.outputs = self.activations[-1]

        # Store model variables for easy access存储模型变量以便于访问
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics建立(评估)指标
        self._loss()       # 调用下边的_loss()
        self._accuracy()   # 调用下边的_accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass
        # pass 是空语句，是为了保持程序结构的完整性。pass 不做任何事情，一般用做占位语句。

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):     # 保存模型函数
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):     # 加载模型函数
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):    # 多层感知机类，继承基础模型类——————————————————————————————
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]
        # To be supported in future Tensorflow versions   上句是为未来的tensorflow版本做准备的
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # 使用Adam优化器

        self.build()

    def _loss(self):
        # Weight decay loss权重衰变损失
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            # l2_loss()这个函数的作用是利用L2范数来计算张量的误差值，但是没有开方并且只取L2范数的值的一半。
            # 一般用于优化的目标函数中的正则项，防止参数太多复杂容易过拟合。

        # Cross entropy error交叉熵
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])
        self.preds = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):
        # 向类的layer列表属性中添加稠密层和读出层
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(ReadoutLayer(input_dim=FLAGS.hidden, output_dim=self.output_dim,
                                        placeholders=self.placeholders, act=lambda x: x,
                                        dropout=True, logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GNN(Model):     # 图神经网络类，继承基础模型类——————————————————————————————
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GNN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.mask = placeholders['mask']
        self.placeholders = placeholders

        # 临时注释————2020.10.22**********************************************************************************
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)    # FLAGS.learning_rate_2021.4.27
        # *************************************************************************************************************
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        print('build...')
        self.build()

    def _loss(self):
        # Weight decay loss权重衰变损失
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for var in tf.trainable_variables():
            if 'weights' in var.name or 'bias' in var.name:
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

                # 测试——————2020.10.21
                # print("调用了GNN的_loss()函数")

        # 测试—————2020.10.7
        # print("调用_loss()")

        # Cross entropy error交叉熵
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])
        self.preds = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):
        
        self.layers.append(GraphLayer(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden,
                                      placeholders=self.placeholders,
                                      act=tf.tanh,
                                      sparse_inputs=False,
                                      dropout=True,
                                      steps=FLAGS.steps,
                                      logging=self.logging))

        self.layers.append(ReadoutLayer(input_dim=FLAGS.hidden,
                                        output_dim=self.output_dim,
                                        placeholders=self.placeholders,
                                        act=tf.tanh,
                                        sparse_inputs=False,
                                        dropout=True,
                                        logging=self.logging))
        
    def predict(self):
        return tf.nn.softmax(self.outputs)
