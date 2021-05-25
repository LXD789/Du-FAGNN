import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import re
from tqdm import tqdm
import jieba
# import sparse


def parse_index_file(filename):
    """Parse index file.解析索引文件"""
    index = []     # 定义列表index
    for line in open(filename):   # 对于打开的文件filename中的每一行(每一次循环)，
        #                            在index中添加这一行去掉字符串头尾字符后的新字符串。
        index.append(int(line.strip()))
    # strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。返回移除字符串头尾指定的字符生成的新字符串。
    # 该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
    return index     # 返回index，即返回文件中在每一行去掉字符串头尾后的内容


def sample_mask(idx, l):
    """Create mask.生成掩码"""
    mask = np.zeros(l)     # 定义n维数组类型(ndarray)对象mask，赋值为有l(英文字母l)个元素的用0填充的数组
    mask[idx] = 1          # 将mask中下标为idx的元素置为1(数字1)
    return np.array(mask, dtype=np.bool)     # 返回数组mask，并将其元素转换为bool型(0转换为false，1转换为true)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory从文件中加载并解析、提取(分离)输入数据

    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
                        训练实例的特征向量和邻接矩阵，为列表
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
                          测试实例的特征向量和邻接矩阵，为列表
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
                           标记和未标记训练实例的特征向量和邻接矩阵（ind.dataset_str.x的超集），为列表
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
                         带标签的训练实例的独热标签，为numpy.ndarray对象
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
                          测试实例的独热标签，为numpy.ndarray对象
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
                            ind.dataset_str.allx中实例的标签，为numpy.ndarray对象
    All objects above must be saved using python pickle module.以上所有的对象必须使用python pickle module保存。
    python pickle module：python的序列化文件保存形式

    :param dataset_str: Dataset name数据集名
    :return: All data input files loaded (as well the training/test data).加载的所有数据输入文件（以及训练/测试数据）。
    """
    names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'allx_adj', 'allx_embed', 'ally']
    # 列表类对象names存储各类文件名
    objects = []     # 定义列表类对象object

    # 1.解析文件(将序列化的文件内容进行反序列化)
    for i in range(len(names)):     # 对于每次循环里names中的每一个元素：
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            # 此句意为打开相应文件，并将其标记为f
            # “data/ind.{}.{}”文件中的{}{}用dataset_str和当前names中的元素来表示
            # format()：格式化字符串的函数str.format()，它增强了字符串格式化的功能。基本语法是通过{}和:来代替以前的%。
            if sys.version_info > (3, 0):
                # 此句：若python版本大于3.0，则在列表objects中添加元素(新添元素为当前打开并已做前边处理后的文件)，
                #       编码方式为latin1
                # sys.version_info：获取python版本号(此程序使用的是3.6)
                objects.append(pkl.load(f, encoding='latin1'))
                # pkl.load():反序列化对象，将文件中的数据解析为一个python对象。
            else:     # 否则在objects中添加元素(当前打开并已做前边处理后的文件)，编码方式为函数默认方式
                objects.append(pkl.load(f))

    x_adj, x_embed, y, tx_adj, tx_embed, ty, allx_adj, allx_embed, ally = tuple(objects)

    # 将objects(列表类对象)转换为元组类对象，并赋值给以上9个变量(即objects中对应的9个部分分别赋给这9个变量)
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    train_adj = []     # 训练集的邻接矩阵
    train_embed = []   # 训练集的特征向量
    val_adj = []       # 验证集的邻接矩阵
    val_embed = []     # 验证集的特征向量
    test_adj = []      # 测试集的邻接矩阵
    test_embed = []    # 测试集的特征向量
    # 定义6个列表类对象

    # 2.分别将训练集、测试集和验证集的邻接矩阵和特征向量提取出来单独存放为列表，并将他们转换为数组。
    # 下面这个for循环是将训练集的邻接矩阵和特征向量分别提取出来(为什么多此一举？答：元组不可更改，所以要转换为数组)
    for i in range(len(y)):           # 对于循环里文件y(带标签的训练实例的独热标签，为numpy.ndarray对象)中的每一个元素：
        adj = x_adj[i].toarray()       # 定义adj，赋值为转换成数组类型的x_adj的当前元素
        embed = np.array(x_embed[i])   # 定义embed，赋值为转换成数组类型的x_embed的当前元素
        train_adj.append(adj)          # 在train_adj中添加adj(即x_adj当前元素)
        train_embed.append(embed)      # 在train_embed中添加embed(即x_embed当前元素)

    # 下面这个for循环是将训练集的超集的邻接矩阵和特征向量分别提取出来，放入各自的列表中
    for i in range(len(y), len(ally)):  # train_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        val_adj.append(adj)
        val_embed.append(embed)

    # 下面这个for循环是将测试集的特征向量和邻接矩阵分别提取出来，放入各自的列表中
    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        test_adj.append(adj)
        test_embed.append(embed)

    # 将列表转换为数组(这里不明白为什么要将列表转换为数组)
    train_adj = np.array(train_adj)
    val_adj = np.array(val_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)

    train_y = np.array(y)   # 将“带标签的训练集的独热标签”转换为数组，存入train_y中
    val_y = np.array(ally[len(y):len(ally)])  # train_size])
    # 将allx中有标记的那部分训练集数据的标签的“从‘y的元素个数’到‘ally的元素个数’”这部分数据转换为数组存入val_y中
    test_y = np.array(ty)

    # 测试——2020.9.15
    # print("ally样子：", ally)
    # print("val_y样子：", val_y)

    return train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.将稀疏矩阵转换为元组表示形式。"""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):     # 若mx为非coo_matrix:
            # isspmatrix_coo(x)：判断x是否为coo_matrix，返回值为bool。(若是，返回true；若否，返回false)

            # coo_matrix：coo_matrix((data, (i, j)), [shape=(M, N)])参数：
            # data[:] 就是原始矩阵中的数据，例如上面的4,5,7,9；
            # i[:] 就是行的指示符号；例如上面row的第0个元素是0，就代表data中第一个数据在第0行；
            # j[:] 就是列的指示符号；例如上面col的第0个元素是0，就代表data中第一个数据在第0列；
            # shape参数是告诉coo_matrix原始矩阵的形状，除了上述描述的有数据的行列，其他地方都按照shape的形式补0。

            mx = mx.tocoo()     # 则令mx重新赋值为coo_matrix的形式
            # tocoo():返回稀疏矩阵的coo_matrix形式
            # 主要用来创建矩阵，因为coo_matrix无法对矩阵的元素进行增删改等操作。
            # 一旦创建之后，除了将之转换成其它格式的矩阵，几乎无法对其做任何操作和矩阵运算。
        coords = np.vstack((mx.row, mx.col)).transpose()

        # 定义数组coords，将mx.row和mx.col竖向堆叠，并调换行列值的索引值
        # vstack()：将数组竖向堆叠(即把几个数组竖着堆起来)
        # transpose():调换数组的行列值的索引值(如二维数组中，行列是按照(x,y)即(0,1)顺序来的，调换后为(y,x)即(1,0))

        values = mx.data
        shape = mx.shape
        # 测试
        # print("coords:", coords)
        # print("values:", values)
        # print("shape:", shape)
        return coords, values, shape   # 返回构成稀疏矩阵的行索引和列索引组成的矩阵、稀疏矩阵的原矩阵和稀疏矩阵的行列值

    if isinstance(sparse_mx, list):    # isinstance():返回对象是类的实例还是子类的实例
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])   # 转换为元组
            # 测试
            # print("sparse_mx[i]:", sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)   # 转换为元组
        # print("sparse_mx:", sparse_mx)

    return sparse_mx   # 返回的spars_mx为元组类型
    # 2020.9.18测试：打印sparse_mx结果:
    # (array([[0, 0],
    #        [0, 1],
    #        [0, 2],
    #        [0, 3],
    #        [0, 4]], dtype=int32), array([ 3,  8,  2,  3, 10]), (1, 5))


def coo_to_tuple(sparse_coo):
    return sparse_coo.coords.T, sparse_coo.data, sparse_coo.shape
    # 将coo_matrix(对角存储矩阵)类型转换为tuple(元组)类型
    # 自我猜测：返回矩阵的转置、矩阵的内容和矩阵的形状


def preprocess_features(features):   # features：特征矩阵组成的总矩阵
    """Row-normalize feature matrix and convert to tuple representation行归一化特征矩阵并转换为元组表示"""
    max_length = max([len(f) for f in features])
    
    for i in tqdm(range(features.shape[0])):    # features.shape[0]：总特征矩阵的一行(即一个特征矩阵)
        feature = np.array(features[i])         # 取feature=features的一行(一个特征矩阵)，转换为数组

        # 测试——————2020.10.22
        # print("feature:", feature)

        pad = max_length - feature.shape[0]  # padding for each epoch为每个epoch做填充
        # 定义pad为max_length与feature的行数的差值

        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')  # 这个地方有错误&*&*&*&*&*&**&*&*&*&*&*&*&*&*&
        # 此句：对feature(即当前的一个特征矩阵)进行填充。
        # numpy.pad(n, pad_width=((2, 3), (3, 3)), mode='constant')参数：
        # n:代表的是数组
        # pad_width：代表的是不同维度填充的长度，（2，3）分别代表第一个维度左填充2，右边填充3。
        #           （3，3）代表第二个维度左边填充3右边填充3。
        #            第一个维度：指的是第一个括号内的元素。第二个维度：指的是第二个括号内的元素。第n个维度：依次类推。

        features[i] = feature   # 将features的当前元素(当前的一个特征矩阵)重新赋值为feature
    
    return np.array(list(features))   # 返回数组类型的features(特征矩阵)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.对称归一化邻接矩阵"""
    # 归一化作用：将数据规范到[0,1]中，以便处理。
    rowsum = np.array(adj.sum(1))    # 计算矩阵adj每一行元素相加之和(axis=1)，将结果转换为数组类型，赋值给rowsum
    # axis=0：表示纵轴，进行操作的方向为从上到下
    # axis=1：表示横轴，进行操作的方向为从左到右
    with np.errstate(divide='ignore'):
        # errstate()：用于浮点错误处理的上下文管理器。
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        # 定义d_inv_sqrt，赋值为rowsum的-0.5次方，并对所得结果进行降维(降到一维)。最后结果(d_inv_sqrt)为数组类型。
        # 1.power(x, y)函数，计算x的y次方。
        # 2.flatten是numpy.ndarray.flatten的一个函数，即返回一个一维数组。
        #   flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用。默认是按行的方向降维。
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # 此句表示：溢出部分赋值为0
    # np.isinf()：判断括号中的内容是否为正负无穷，返回bool类型。(若是，返回true；否则返回false)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    # 此句意为：定义数组d_mat_inv_sqrt，其接收d_inv_sqrt经np.diag()(即d_inv_sqrt对角化)的结果。
    # np.diag():以一维数组的形式返回方阵的对角线（或非对角线）元素，或将一维数组转换成方阵（非对角线元素为0）。
    #   两种功能角色转变取决于输入的v。
    # 参数：
    # v : array_like.
    # 如果v是2D数组，返回k位置的对角线。如果v是1D数组，返回一个v作为k位置对角线的2维数组。
    # k : int, optional。对角线的位置，大于零位于对角线上面，小于零则在下面。
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    # 此句意为：将adj与d_mat_inv_sqrt做点乘，再将其进行转置，然后再与d_mat_inv_sqrt进行点乘。返回最后的结果(数组类型)。


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
       简单GCN模型的邻接矩阵预处理并转换为元组表示。"""
    max_length = max([a.shape[0] for a in adj])
    # 方括号中的for循环称为列表解析。与单独的for循环等价，但速度快一倍。for前面的相当于for循环内的操作。
    # 此句意为：在循环中将adj里的每个元素(元素为矩阵)的行数提出来单独组成一个列表，并在这个列表中寻找最大值(即最大行数)，
    #           并把最大行数赋值给max_length
    mask = np.zeros((adj.shape[0], max_length, 1))  # mask for padding
    # 定义数组mask，赋值为有“adj行数”个矩阵、max_length行、1(数字1)列的全为零的数组(三维)

    for i in tqdm(range(adj.shape[0])):
        # tqdm：一个快速，可扩展的Python进度条，可以在Python长循环中添加一个进度提示信息。
        adj_normalized = normalize_adj(adj[i])  # no self-loop
        # 将当前adj元素进行对称归一化
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        # 定义pad，赋值为max_length(最大行数)与adj_normalized的行数之差
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        # 对数组adj_normalized进行填充。
        mask[i, :adj[i].shape[0], :] = 1.
        # 数组的“第一维的当前元素，第二维从开始位置到adj[i].shape[0](adj当前元素的行数)，第三维的全部”以上元素赋值为1
        adj[i] = adj_normalized   # adj的当前元素重新赋值为adj_normalized

    return np.array(list(adj)), mask  # coo_to_tuple(sparse.COO(np.array(list(adj)))), mask
    # 返回转换为数组的adj(邻接矩阵)和数组mask(掩码数组)


def construct_feed_dict(features, support, mask, labels, placeholders):
    """Construct feed dictionary.构造提要字典（将占位符进行赋值）"""
    # feed_dict作用：给使用placeholder创建出来的tensor赋值。
    feed_dict = dict()     # 定义feed_dict为字典类对象
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['mask']: mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
       计算Chebyshev多项式直至k。 返回稀疏矩阵的列表（元组表示）。"""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors读取词向量"""
    vocab = []               # 词汇列表
    embd = []                # 嵌入列表
    word_vector_map = {}     # 词向量字典
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')  # 对一行文本先进行去头尾字符串操作、再通过空格进行文本分割(分词)，返回的是列表
        if len(row) > 2:     # 若这一行文本超过两个词：
            vocab.append(row[0])     # 在vocab(词汇列表)中添加row列表的第一个元素
            vector = row[1:]         # 定义列表vector，赋值为row列表从第二个元素开始到末尾的内容
            length = len(vector)     # 定义length，赋值为为列表vector中含有的元素个数
            for i in range(length):
                vector[i] = float(vector[i])   # 类型转换（string——>float）
            embd.append(vector)       # 在embd(嵌入列表)中添加vector列表(中的元素)
            word_vector_map[row[0]] = vector   # 将字典word_vector_map中的键“row[0]”对应的值赋为vector
    print('Loaded Word Vectors!')
    file.close()   # 关闭文件
    return vocab, embd, word_vector_map    # 返回词汇列表、嵌入列表和词向量字典


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.除SST外，所有数据集的标记化/字符串清除。
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # re：正则表达式。re.sub()：正则表达式替换函数。sub即substitute(替换)
    # sub(pattern, repl, string, count=0, flags=0)参数:
    # （1）pattern：该参数表示正则中的模式字符串；
    # （2）repl：该参数表示要替换的字符串（即匹配到pattern后替换为repl），也可以是个函数；
    # （3）string：该参数表示要被处理（查找替换）的原始字符串；
    # （4）count：可选参数，表示是要替换的最大次数，而且必须是非负整数，该参数默认为0，即所有的匹配都会被替换；
    # （5）flags：可选参数，表示编译时用的匹配模式（如忽略大小写、多行模式等），数字形式，默认为0。

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # 那些看起来替换前后无变化的语句，实际上在替换后都在新符号前多了一个空格(加空格的意义：方便下一步操作，即方便分词)

    return string.strip().lower()
    # lower():转换大写字符为小写字符


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset.SST数据集的标记化/字符串清除。
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


# def clean_ch_str(string):   # 清洗中文数据

# load_data("mr")   # 测试——2020.9.16

# row = [1, 2, 3, 3, 2]
# col = [1, 3, 4, 2, 3]
# data = [3, 8, 2, 3, 10]
# c = sp.coo_matrix(data)
# # print("c1:", c)
# sparse_to_tuple(c)
# print("c2:", c)
# 测试——2020.9.18

