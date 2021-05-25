# 如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，
# 也就是说它在当前版本中不是语言标准，那么我们如果想要使用的话就要从__future__模块导入
from __future__ import print_function  # print()函数

import scipy.sparse as sp  # python中稀疏矩阵相关库
import numpy as np  # python中操作数组的函数
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence  # 稀疏矩阵中查找特征值/特征向量的函数
# import json


# 将标签转换为one-hot编码形式
def encode_onehot(labels):
    # set()函数创建一个不重复元素集合
    classes = set(labels)
    # np.identity()函数创建方针，返回主对角线元素为1，其余元素为0的数组
    # enumerate()函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列
    # 同时列出数据和数据下标，一般用在for循环中
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # map()函数根据提供的函数对指定序列做映射
    # map(function, iterable)
    # 第一个参数function以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# 加载数据
def load_data0(path="record/", dataset=""):   # ————————————————2021.4.2
    """Load citation network dataset (cora only for now)"""
    # str.format()函数用于格式化字符串
    # print('Loading event {} ...'.format(dataset))
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype, delimiter, usecols, skip_header)
    # frame：文件名
    # dtype：数据类型
    # delimiter：分隔符
    # usecols：选择读哪几列，通常将属性集读为一个数组，将标签读为一个数组
    # skip_header：是否跳过表头
    idx_features_labels = np.genfromtxt("{}{}_check.txt".format(path, dataset), dtype=np.dtype(str))
    # print("idx_features_labels:", idx_features_labels)
    if dataset != "3911188850873165" and dataset != "3586313934109778":
        # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        # print("feature行数：", features.shape[0])  # ----->10
        # print("feature列数：", features.shape[1])  # ----->256

        # 提取样本的标签，并将其转换为one-hot编码形式
        labels = []
        for i1 in range(len(idx_features_labels)):
            labels.append(idx_features_labels[i1][-1])
        # labels = idx_features_labels[:, -1]
        # labels = encode_onehot(idx_features_labels[:, -1])
        # print("label行数：", labels.shape[0])
        # print("label列数：", labels.shape[1])

        # 样本的id数组
        idx = np.array(idx_features_labels[:, 0], dtype=np.int64)
        # print("idx:", idx)
    else:
        # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
        features = sp.csr_matrix(idx_features_labels[1:-1], dtype=np.float32)
        # print("feature行数：", features.shape[0])  # ----->10
        # print("feature列数：", features.shape[1])  # ----->256

        # 提取样本的标签，并将其转换为one-hot编码形式
        labels = []
        for i1 in range(len(idx_features_labels)):
            labels.append(idx_features_labels[i1][-1])
        # labels = encode_onehot(idx_features_labels[-1])
        # print("label行数：", labels.shape[0])
        # print("label列数：", labels.shape[1])

        # 样本的id数组
        idx = np.array(idx_features_labels[0], dtype=np.int64)
        # print("idx:", idx)

    return features.todense(), labels, idx


# feat1, lab1, idx1 = load_data0(dataset="3911194089597191")   # 3911188850873165
# print("idx.type:", type(idx1))
# idx1 = idx1.tolist()
# print("idx.type:", type(idx1))
# print("shape:", idx1.shape)
# idx1.reshape((1, ))
# print("shape1:", idx1.shape)


def load_data_all(path="data/weibo/", dataset="weibo"):   # feature共72261行，edge共68302行
    print('Loading {} dataset...'.format(dataset))

    eid_file = open("eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_file.readlines()  # eid_list

    for i in range(len(eid_list)):
        eid_list[i] = eid_list[i].replace("\n", "")

    feat_all = None
    # label_all = None
    label_all = []
    idx_all = None
    for eid in eid_list:
        feat, label, idx = load_data0(dataset=eid)
        # idx = idx.tolist()
        if eid_list.index(eid) == 0:
            feat_all = feat
            label_all.extend(label)
            # label_all = label
            idx_all = idx
        else:
            feat_all = np.concatenate((feat_all, feat))
            # label_all = np.concatenate((label_all, label))
            label_all.extend(label)
            if eid != "3911188850873165" and eid != "3586313934109778":
                idx_all = np.concatenate((idx_all, idx))
            else:
                idx_all = np.concatenate((idx_all, [idx]))

    label_all = encode_onehot(label_all)
    print("label_all.shape:", label_all.shape)

    # idx_all = np.array(idx_all)
    # 由样本id到样本索引的映射字典
    idx_map = {j: i for i, j in enumerate(idx_all)}  # i为索引，j为样本id。所以该字典为：{样本id:样本索引}
    # print("idx_map:", idx_map)

    # ss = open("idx_map.txt", "w", encoding="utf-8")
    # ss.write(str(idx_map))

    # 样本之间的引用关系数组
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int64)
    # print("edges_unordered行数：", edges_unordered.shape[0])
    # print("edges_unordered列数：", edges_unordered.shape[1])
    # print("edges_unordered[:, 0]:", edges_unordered[:, 0])
    # print("edges_unordered[:, 1]:", edges_unordered[:, 1])

    # 将样本之间的引用关系用样本索引之间的关系表示——————————>即：将样本id换成其对应的样本索引
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    # print("edges.type:", type(edges))  # ndarray
    # print("edges行数：", edges.shape[0])
    # print("edges列数：", edges.shape[1])

    # ww = open("edges.txt", "w", encoding="utf-8")
    # ww.write(str(list(edges)))

    # edges[:, 0] [0 0 0 0 0 0 0 0 0]
    # edges[:, 1] [1 2 3 4 5 6 7 8 9]
    # print("edges[:, 0]:", edges[:, 0])
    # print("edges[:, 0].shape:", edges[:, 0].shape)
    # print("edges[:, 1]:", edges[:, 1])
    # print("edges[:, 1].shape:", edges[:, 0].shape)
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(label_all.shape[0], label_all.shape[0]), dtype=np.float32)
    # print("adj行数：", adj.shape[0])
    # print("adj列数：", adj.shape[1])
    print("old_adj:", adj.shape)

    # build symmetric adjacency matrix
    # 将非对称邻接矩阵转变为对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print("feature:", feat_all.shape)
    print("adj:", adj.shape)
    print("edges:", edges.shape)

    # 打印消息：数据集有多少个节点、多少条边、每个样本有多少维特征
    print('Dataset has {} nodes, {} edges, {} dim of features.'.format(adj.shape[0], edges.shape[0], feat_all.shape[1]))
    # Dataset has 72380 nodes, 67716 edges, 256 dim of features.
    # Dataset has 72894 nodes, 67716 edges, 256 dim of features.
    return feat_all, adj, label_all

# edges行数： 67716
# adj行数、列数： 72380


# load_data_all()


def load_data(path="", dataset="weibo"):
    """Load citation network dataset (cora only for now)"""
    # str.format()函数用于格式化字符串
    print('Loading {} dataset...'.format(dataset))
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype, delimiter, usecols, skip_header)
    # frame：文件名
    # dtype：数据类型
    # delimiter：分隔符
    # usecols：选择读哪几列，通常将属性集读为一个数组，将标签读为一个数组
    # skip_header：是否跳过表头
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # print("idx_features_labels:", idx_features_labels)

    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = np.matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    print("features:", features.shape)
    #
    # 提取样本的标签，并将其转换为one-hot编码形式
    labels = encode_onehot(idx_features_labels[:, -1])
    print("labels：", labels.shape)
    #
    # build graph
    # 样本的id数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int64)

    # 由样本id到样本索引的映射字典
    idx_map = {j: i for i, j in enumerate(idx)}  # i为索引，j为样本id。所以该字典为：{样本id:样本索引}
    # print("idx_map:", idx_map)
    # 样本之间的引用关系数组
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int64)
    print("edges_unordered：", edges_unordered.shape)
    #
    # 将样本之间的引用关系用样本索引之间的关系表示——————————>即：将样本id换成其对应的样本索引
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    # print("edges.type:", type(edges))  # ndarray
    print("edges：", edges.shape)
    #
    # edges[:, 0] [0 0 0 0 0 0 0 0 0]
    # edges[:, 1] [1 2 3 4 5 6 7 8 9]
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    print("adj：", adj.shape)
    # ww = open("adj.txt", "w", encoding="utf-8")
    # ww.write(str(adj))
    #
    # build symmetric adjacency matrix
    # 将非对称邻接矩阵转变为对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 打印消息：数据集有多少个节点、多少条边、每个样本有多少维特征
    print('Dataset has {} nodes, {} edges, {} dim of features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    # Dataset has 2708 nodes, 5429 edges, 1433 dims of features.
    # 返回特征的密集矩阵表示、邻接矩阵和标签的one-hot编码

    # features: (72380, 256)
    # labels： (72380, 2)
    # edges_unordered： (67716, 2)
    # edges： (67716, 2)
    # adj： (72380, 72380)
    # Dataset has 72380 nodes, 67716 edges, 256 dim of features.

    return features, adj, labels


# load_data()


# 对邻接矩阵进行归一化处理
def normalize_adj(adj, symmetric=True):
    # 如果邻接矩阵为对称矩阵，得到对称归一化邻接矩阵
    # D^(-1/2) * A * D^(-1/2)
    if symmetric:
        # A.sum(axis=1)：计算矩阵的每一行元素之和，得到节点的度矩阵D
        # np.power(x, n)：数组元素求n次方，得到D^(-1/2)
        # sp.diags()函数根据给定的对象创建对角矩阵，对角线上的元素为给定对象中的元素
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        # tocsr()函数将矩阵转化为压缩稀疏行矩阵
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    # 如果邻接矩阵不是对称矩阵，得到随机游走正则化拉普拉斯算子
    # D^(-1) * A
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


# 在邻接矩阵中加入自连接
def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    # 对加入自连接的邻接矩阵进行对称归一化处理
    adj = normalize_adj(adj, symmetric)
    return adj


# 构造样本掩码
def sample_mask(idx, l):   # 不是很清楚为什么要构造样本掩码
    """
    :param idx: 有标签样本的索引列表
    :param l: 所有样本数量
    :return: 布尔类型数组，其中有标签样本所对应的位置为True，无标签样本所对应的位置为False
    """
    # np.zeros()函数创建一个给定形状和类型的用0填充的数组
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# 数据集划分
def get_splits(y):        # 需要修改————————————————————————————————
    # 训练集索引列表
    idx_train = range(57904)
    # idx_train = range(140)
    # 验证集索引列表
    idx_val = range(57904, 65142)
    # idx_val = range(200, 500)
    # 测试集索引列表
    idx_test = range(65142, 72380)  # 300
    # idx_test = range(500, 1500)
    # 训练集样本标签
    y_train = np.zeros(y.shape, dtype=np.int32)
    # 验证集样本标签
    y_val = np.zeros(y.shape, dtype=np.int32)
    # 测试集样本标签
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    # 训练数据的样本掩码
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


# 定义分类交叉熵
def categorical_crossentropy(preds, labels):   # 损失函数
    """
    :param preds:模型对样本的输出数组
    :param labels:样本的one-hot标签数组
    :return:样本的平均交叉熵损失
    """
    # np.extract(condition, x)函数，根据某个条件从数组中抽取元素
    # np.mean()函数默认求数组中所有元素均值
    return np.mean(-np.log(np.extract(labels, preds)))


# 定义准确率函数
def accuracy(preds, labels):
    # np.argmax(x)函数取出x中元素最大值所对应的索引
    # np.equal(x1, x2)函数用于在元素级比较两个数组是否相等
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


# 评估样本划分的损失函数和准确率
def evaluate_preds(preds, labels, indices):
    """
    :param preds:对于样本的预测值
    :param labels:样本的标签one-hot向量
    :param indices:样本的索引集合
    :return:交叉熵损失函数列表、准确率列表
    """
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        # 计算每一个样本划分的交叉熵损失函数
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        # 计算每一个样本划分的准确率
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


# 对拉普拉斯矩阵进行归一化处理
def normalized_laplacian(adj, symmetric=True):
    # 对称归一化的邻接矩阵，D ^ (-1/2) * A * D ^ (-1/2)
    adj_normalized = normalize_adj(adj, symmetric)
    # 得到对称规范化的图拉普拉斯矩阵，L = I - D ^ (-1/2) * A * D ^ (-1/2)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


# 重新调整对称归一化的图拉普拉斯矩阵，得到其简化版本
def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        # 计算对称归一化图拉普拉斯矩阵的最大特征值
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    # 如果计算过程不收敛
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    # 调整后的对称归一化图拉普拉斯矩阵，L~ = 2 / Lambda * L - I
    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


# 计算直到k阶的切比雪夫多项式
def chebyshev_polynomial(X, k):
    # 返回一个稀疏矩阵列表
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())  # T0(X) = I
    T_k.append(X)  # T1(X) = L~

    # 定义切比雪夫递归公式
    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        """
        :param T_k_minus_one: T(k-1)(L~)
        :param T_k_minus_two: T(k-2)(L~)
        :param X: L~
        :return: Tk(L~)
        """
        # 将输入转化为csr矩阵（压缩稀疏行矩阵）
        X_ = sp.csr_matrix(X, copy=True)
        # 递归公式：Tk(L~) = 2L~ * T(k-1)(L~) - T(k-2)(L~)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    # 返回切比雪夫多项式列表
    return T_k


# 将稀疏矩阵转化为元组表示
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        # 将稀疏矩阵转化为coo矩阵形式
        # coo矩阵采用三个数组分别存储行、列和非零元素值的信息
        sparse_mx = sparse_mx.tocoo()
    # np.vstack()函数沿着数组的某条轴堆叠数组
    # 获取非零元素的位置索引
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # 获取矩阵的非零元素
    values = sparse_mx.data
    # 获取矩阵的形状
    shape = sparse_mx.shape
    return coords, values, shape
