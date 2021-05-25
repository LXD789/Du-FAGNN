from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
# tf中定义了tf.app.flags.FLAGS，用于接受从终端传入的命令行参数

# global unique layer ID dictionary for layer name assignment用于分配层名称的全局唯一层ID字典，
_LAYER_UIDS = {}    # _LAYER_UIDS为字典类对象


def get_layer_uid(layer_name=''):     # 获取层的ID
    """Helper function, assigns unique layer IDs.帮助函数，分配唯一层ID"""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
        # 若layer_name不在_LAYER_UIDS中，则字典_LAYER_UIDS里的键“layer_name”对应的值为1，且返回1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]
        # 否则(即若layer_name在_LAYER_UIDS中)字典_LAYER_UIDS里的键“layer_name”对应的值自加1


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors.丢弃稀疏张量"""
    random_tensor = keep_prob     # 定义张量类对象random_tensor，赋值为元素保留概率
    # keep_prob：与x具有相同类型的标量张量，意为保留每个元素的概率。
    # 即每个元素被保留的概率，则keep_prob:1就是所有元素全部保留的意思。
    # 一般在大量数据训练时，为了防止过拟合，添加Dropout层，设置一个0~1之间的小数。

    random_tensor += tf.random_uniform(noise_shape)
    # 此句意为：random_tensor重新赋值为原random_tensor+(noise_shape*noise_shape)的矩阵

    # 一般情况下的noise_shape:
    # noise_shape是一个一维张量，其实就是一个一维数组（可以是list或者tuple），长度必须跟x.shape一样。
    # 而且noise_shape里边的元素，只能是1或者是x.shape里边对应的元素。哪个轴为1，哪个轴就会被一致地dropout。
    # （可以理解为，加在每个样本的噪音都是一样的。）被丢弃的整行或者整列元素都为0。
    # 自己的理解：noise_shape即噪音张量，是dropout过程中用于表示丢弃哪些元素(或哪几维度的元素)。

    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    # tf.cast():将张量转换为新类型(转换为bool型)
    # tf.floor():返回不大于x的元素最大整数。（即向下取整）

    pre_out = tf.sparse_retain(x, dropout_mask)
    # 此句意为：定义SparseTensor类对象pre_out，赋值为tf.sparse_retain()(保留指定非空值)的返回值
    # sparse_retain():在一个SparseTensor(稀疏张量)中保留指定的非空值。
    # sparse_retain(sp_input,to_retain)参数：
    #    sp_input：输入的SparseTensor，带有N个非空元素。
    #    to_retain：一个布尔类型的向量，向量长度是N，并且其中包含M个True值。
    #    输出：一个SparseTensor，数据维度和输入数据相同，其中包含M个非空值，该值的位置根据True的位置来决定。
    return pre_out * (1./keep_prob)    # 返回pre_out与元素保留概率倒数(简单理解为期望)的乘积


def sparse_dense_matmul_batch(sp_a, b):
    """1.对稀疏张量进行重塑后将其与稠密张量做乘积。
       2.调用map_function()，映射从维度0的“elems”解压缩的张量列表(即运行tf.map_fn())"""
    def map_function(x):
        i, dense_slice = x[0], x[1]     # 定义i和dense_slice(稠密片)，分别赋值为x的第一个元素和第二个元素
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        # 定义sparse_slice(稀疏片)，赋值为对sp_a进行从[i,0,0]开始的、
        #     大小为"[1, sp_a.dense_shape[1], sp_a.dense_shape[2]]"的切片的重塑(reshape)

        # tf.sparse.reshape():重塑一个SparseTensor以表示新密集形状中的值。该操作在表示的稠密张量上与reshape具有相同的语义
        # 根据新的密集形状重新计算sp_input中非空值的索引，并返回包含新索引和新形状的新SparseTensor。
        # sp_input中非空值的顺序不变。
        # sparse_reshape(sp_input,shape,name=None)参数：
        #     sp_input：输入的 SparseTensor.
        #     shape：一个 1-D(矢量)int64的Tensor,指定所表示的SparseTensor的新密集形状.
        #     name：返回张量的名称前缀(可选)
        # shape里最多有一个维度的值可以填写为-1，表示自动计算此维度。

        # tf.sparse.slice(sp_input, start, size, name=None)：根据start和size对SparseTensor进行切片。

        mult_slice = tf.sparse.matmul(sparse_slice, dense_slice)
        # 定义mult_slice，赋值为sparse_slice(稀疏矩阵)和dense_slice(稠密矩阵)的乘积
        return mult_slice   # 返回mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    # 此句：定义元组elems(元组是不可变的list)，值为"从0开始、以delta=1的增量扩展、直至但不包括sp_a.dense_shape[0]"和b
    # tf.range(start, limit, delta, dtype):创建一个数字序列，该数字序列从“ start”开始，并以“ delta”的增量扩展，
    # 直至但不包括“ limit”。除非明确提供，否则从输入中推断出结果张量的dtype。

    return tf.map_fn(map_function, elems, dtype=tf.float32, back_prop=True)
    # tf.map_fn():映射从维度0的“elems”解压缩的张量列表。

    # 自我理解：对elems中的每一个张量，都调用map_function()来进行计算。
    #          返回值为elems中每个张量进行map_function()返回的张量或张量序列。
    # map_fn的最简单版本将可调用的fn重复应用于从头到尾的一系列元素。 这些元素由从`elems`中解压缩的张量组成。
    # dtype是fn返回值的数据类型。 如果dtype与elems的数据类型不同，则用户必须提供dtype。


def dot(x, y, sparse=False):
    """Wrapper for 3D tf.matmul (sparse vs dense).3D tf.matmul的装饰器（稀疏vs稠密）。
    装饰器：其作用就是为已经存在的函数或对象添加额外的功能。装饰器本质上是一个Python函数，
    它可以让其他函数在不需要做任何代码变动的前提下增加额外功能，装饰器的返回值也是一个函数对象。"""
    # tf.matmul()：两个矩阵相乘("Aij*Bjk=Cik"的这种)
    if sparse:     # 若张量为稀疏张量，则定义张量res，赋值为sparse_dense_matmul_batch()(上边的函数)的返回值
        res = sparse_dense_matmul_batch(x, y)
    else:     # 否则(即若张量为稠密张量)定义张量res，赋值为张量x和y的广义收缩函数的返回值(自我理解：即两个张量相乘)
        res = tf.einsum('bij,jk->bik', x, y)  # tf.matmul(x, y)
        # tf.einsum():任意维度的张量之间的广义收缩。
        # einsum(equation, *inputs, **kwargs):该函数返回一个张量，其张量由“equation”定义
    return res


def gru_unit(support, x, var, act, mask, dropout, sparse_inputs=False):
    """GRU unit with 3D tensor inputs.带有3D张量输入的GRU单元"""
    # message passing消息传递
    #   1.产生丢弃后的张量
    support = tf.nn.dropout(support, dropout)  # optional可选的
    # 定义张量support，接收tf.nn.dropout()的返回值。
    # tf.nn.dropout():计算dropout。概率为“ keep_prob”，输出按“ 1 / keep_prob”放大的输入元素，
    #                 否则输出“ 0”。 缩放比例使预期总和不变。返回与第一个参数形状相同的张量。
    #   2.计算两张量的乘积(其中一个为dropout后的张量)
    a = tf.matmul(support, x)
    # 定义张量a，接收tf.matmul()的返回值
    # tf.matmul():将第一个(张量型)矩阵参数与第二个(张量型)矩阵参数做乘积，返回与这两个参数类型相同的张量

    # 与文章相结合来看：
    # a：节点从其相邻节点接收的信息
    # x：h(t-1)(上一层的输出状态)
    # w,u,b均为可训练的权重和偏置项(w为与a相乘的参数var['weights_z0']和var['weights_r0']，
    #                             u为与x相乘的参数var['weights_z1']和var['weights_r1']，
    #                             b为var['bias_...'])
    # z和r来确定邻居信息对当前节点嵌入的贡献程度

    # update gate  更新门
    z0 = dot(a, var['weights_z0'], sparse_inputs) + var['bias_z0']
    # 自我理解：a为输入，var['weights_z0']对应的值为更新门权重，二者与sparse_inputs一起作为dot()的参数，意为“w*a”，
    #          var['bias_z0']对应的值为偏置项。所以这句话意为：z0=w*a+b。
    z1 = dot(x, var['weights_z1'], sparse_inputs) + var['bias_z1'] 
    z = tf.sigmoid(z0 + z1)

    # reset gate   复位门
    r0 = dot(a, var['weights_r0'], sparse_inputs) + var['bias_r0']
    r1 = dot(x, var['weights_r1'], sparse_inputs) + var['bias_r1']
    r = tf.sigmoid(r0 + r1)

    # update embeddings    更新嵌入
    h0 = dot(a, var['weights_h0'], sparse_inputs) + var['bias_h0']
    h1 = dot(r*x, var['weights_h1'], sparse_inputs) + var['bias_h1']
    h = act(mask * (h0 + h1))
    
    return h*z + x*(1-z)   # 返回值为当前GRU单元的输出状态ht


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.基础层类。 为所有图层对象定义基本API。
    Implementation inspired by keras (http://keras.io).

    # Properties属性
        name: String, defines the variable scope of the layer.定义层的可变范围
        logging: Boolean, switches Tensorflow histogram logging on/off开启/关闭Tensorflow直方图记录
        vars:
        sparse_inputs:

    # Methods方法
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
            # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
            # 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况

            # **的作用：在函数定义中，收集关键字参数到一个新的字典，并将整个字典赋值给变量kwargs
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            # tf.name_scope():初始化上下文管理器。
            # （1）在某个tf.name_scope()指定的区域中定义的所有对象及各种操作，他们的
            #     “name”属性上会增加该命名区的区域名，用以区别对象属于哪个区域；
            # （2）将不同的对象及操作放在由tf.name_scope()指定的区域中，便于在tensorboard中
            #      展示清晰的逻辑关系图，这点在复杂关系图中特别重要。
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
                # tf.summary.histogram():输出带有直方图的“摘要(总结)”协议缓冲区。
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer.稠密层，继承基础层"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout  为稀疏丢弃而设定的帮助变量(辅助变量)
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            # tf.variable_scope():返回上下文管理器，用于定义创建变量(或层)的操作。
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            # 调用了init.py中的glorot()函数(权重初始化)
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                # 调用了init.py中的zero()函数(生成零矩阵)

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            # 若当前对象的sparse_inputs属性为True，则调用当前,py文件中的sparse_dropout()函数，x赋值为其返回值
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
            # 否则x赋值为tf.nn.dropout()的返回值

        # transform变形(自我理解：矩阵(张量)的相乘操作)
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        # 调用当前.py文件中的dot()函数，output赋值为其返回值(张量/矩阵)
        # 此句相当于"y=wx+b"中的"wx"

        # bias偏置项
        if self.bias:
            output += self.vars['bias']
            # 此句相当于"y=wx+b"中的"+b"。因为前边的transform部分已经有了"wx"，所以这一部分是在其基础上添加偏置项

        return self.act(output)   # 当前类对象的act属性为激活函数(默认为relu，但实际上用的是tanh)


class GraphLayer(Layer):
    """Graph layer.图层，继承基础层"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, steps=2, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.mask = placeholders['mask']
        self.steps = steps

        # helper variable for sparse dropout用于稀疏丢弃的帮助变量
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            # 调用inits.py中的glorot()函数对参数(复位门权重、更新门权重和更新嵌入的权重)进行初始化赋值
            self.vars['weights_encode'] = glorot([input_dim, output_dim], name='weights_encode')
            self.vars['weights_z0'] = glorot([output_dim, output_dim], name='weights_z0')
            self.vars['weights_z1'] = glorot([output_dim, output_dim], name='weights_z1')
            self.vars['weights_r0'] = glorot([output_dim, output_dim], name='weights_r0')
            self.vars['weights_r1'] = glorot([output_dim, output_dim], name='weights_r1')
            self.vars['weights_h0'] = glorot([output_dim, output_dim], name='weights_h0')
            self.vars['weights_h1'] = glorot([output_dim, output_dim], name='weights_h1')

            # 调用inits.py中的zeros()函数对参数(偏置项)进行赋值
            self.vars['bias_encode'] = zeros([output_dim], name='bias_encode')
            self.vars['bias_z0'] = zeros([output_dim], name='bias_z0')
            self.vars['bias_z1'] = zeros([output_dim], name='bias_z1')
            self.vars['bias_r0'] = zeros([output_dim], name='bias_r0')
            self.vars['bias_r1'] = zeros([output_dim], name='bias_r1')
            self.vars['bias_h0'] = zeros([output_dim], name='bias_h0')
            self.vars['bias_h1'] = zeros([output_dim], name='bias_h1')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            # 若类对象的sparse_inputs属性为True，则将x赋值为当前.py文件的sparse_dropout()的返回值
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
            # 否则将x赋值为dropout()的返回值

        # encode inputs编码输入
        x = dot(x, self.vars['weights_encode'], 
                self.sparse_inputs) + self.vars['bias_encode']
        # 调用当前.py文件中的dot()函数，x重新赋值为其返回值(dot()的输入为原x，结果为新x)
        # 此句相当于"y=wx+b"
        output = self.mask * self.act(x)
        # 将上边的x输入激活函数后所得的结果与类对象的mask属性相乘，将结果赋值给output

        # convolve卷积
        for _ in range(self.steps):
            output = gru_unit(self.support, output, self.vars, self.act,
                              self.mask, 1-self.dropout, self.sparse_inputs)
            # 循环中执行：调用此.py文件中的gru_unit()函数，原output作为其中一个输入，输出新output

        return output   # 返回output


class ReadoutLayer(Layer):
    """Graph Readout Layer.图读出层，继承基础层"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        super(ReadoutLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.mask = placeholders['mask']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_att'] = glorot([input_dim, 1], name='weights_att')
            self.vars['weights_emb'] = glorot([input_dim, input_dim], name='weights_emb')
            self.vars['weights_mlp'] = glorot([input_dim, output_dim], name='weights_mlp')

            self.vars['bias_att'] = zeros([1], name='bias_att')
            self.vars['bias_emb'] = zeros([input_dim], name='bias_emb')
            self.vars['bias_mlp'] = zeros([output_dim], name='bias_mlp')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # 下边这些目前无法与文章所写相对应，有待进一步思考。2020.9.27————————————————————————
        # soft attention软注意
        att = tf.sigmoid(dot(x, self.vars['weights_att']) + self.vars['bias_att'])
        # 对应文章中的f1(软注意权重)，此句相当于"sigmoid(wx+b)"。
        emb = self.act(dot(x, self.vars['weights_emb']) + self.vars['bias_emb'])
        # 对应文章中的f2(非线性特征变换)，此处的act在实际应用中是tanh激活函数。此句相当于"tanh(wx+b)"

        N = tf.reduce_sum(self.mask, axis=1)
        # tf.reduce_sum():用于计算张量tensor沿着某一维度的和，可以在求和后降维。
        # 此句意为：定义N，赋值为“对类对象的mask属性进行竖向(行方向或称第二个维度)的元素进行求和”后的结果。
        M = (self.mask-1) * 1e9
        # “1e9”表示：1×10^9(自我猜测：1×10^9此处表示正无穷)
        # 定义M为“类对象的mask属性-1后与1×10^9相乘”的结果
        
        # graph summation图总结
        g = self.mask * att * emb
        # 定义g为类对象属性mask与上边的att、emb相乘(意为将att与emb相乘，将结果掩码后赋值给g)
        g = tf.reduce_sum(g, axis=1) / N + tf.reduce_max(g + M, axis=1)
        # tf.reduce_max()：计算tensor指定轴方向上的各个元素的最大值。
        # 此处为沿竖向(行方向或称第二维度)的元素求最大值。
        # 此句相当于文章中的“hG=(h1+h2+...+hn)/n+maxpooling(h1,h2,...,hn)”(但是不明白为什么要加上M)
        g = tf.nn.dropout(g, 1-self.dropout)
        # 进行dropout操作(此处的类对象的dropout属性为元素丢弃率，因此需要用1减去元素丢弃率得到元素保留概率(keep_prob))

        # classification分类
        output = tf.matmul(g, self.vars['weights_mlp']) + self.vars['bias_mlp']
        # 将上边的新的g与mlp的权重相乘并加上mlp的偏置项。即“y=gw+b”的操作。

        # 此处不太明白：为什么只有“y=gw+b”,而没有“softmax（gw+b）”？
        # 文章中还有一步，是将“w*hG(即上边代码里的g)+b”输入到softmax中进行预测标签。这一步在代码中没有体现。
        # 自我解释：这一部分有可能在模型中有写，而不是在这个地方。
        # ——————2020.9.28
        # 对以上疑问的解答：在metrics.py中有softmax

        return output    # 返回输出结果

