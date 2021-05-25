import keras
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.contrib.opt import AdamWOptimizer
from keras.regularizers import l2
from keras.backend import get_session
import tensorflow as tf

import h5py
import yaml
from keras.models import model_from_yaml
from keras.models import load_model

from graph import GraphConvolution, GraphConvolution1
from utils import *
import time

# f = h5py.File("model_weights.h5", "r")
# for k, v in f.items():
#     print("k.type:", type(k))
#     print("v.type:", type(v))


# 超参数
# Define parameters
# 数据集
DATASET = 'weibo'
# 过滤器
FILTER = 'localpool'  # 'chebyshev'
# 最大多项式的度
MAX_DEGREE = 2  # maximum polynomial degree
# 是否对称正则化
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
# 迭代次数
# NB_EPOCH = 20000
# NB_EPOCH = 300
# 提前停止参数
# PATIENCE = 10  # early stopping patience

# 加载数据
# Get data
X, A, y = load_data(dataset=DATASET)  # 特征、邻接矩阵、标签
# 训练集样本标签、验证集样本标签、测试集样本标签、训练集索引列表
# 验证集索引列表、测试集索引列表、训练数据的样本掩码
# y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

# 对特征进行归一化处理
# Normalize X
X /= X.sum(1).reshape(-1, 1)   # X.sum(1).reshape(-1, 1):按行进行相加，然后重新排列成n行1列的矩阵
print("X.shape:", X.shape)  # ——————————————————————————————————————————————

# 当过滤器为局部池化过滤器时
if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    # 加入自连接的邻接矩阵
    A_ = preprocess_adj(A, SYM_NORM)
    print("A_.shape:", A_.shape)  # ———————————————————————————————————————————
    support = 1
    # 特征矩阵和邻接矩阵
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]  # Layer名是input_1

# 当过滤器为切比雪夫多项式时
elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    # 对拉普拉斯矩阵进行归一化处理，得到对称规范化的拉普拉斯矩阵
    L = normalized_laplacian(A, SYM_NORM)
    # 重新调整对称归一化的图拉普拉斯矩阵，得到其简化版本
    L_scaled = rescale_laplacian(L)
    # 计算直到MAX_DEGREE阶的切比雪夫多项式
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    #
    support = MAX_DEGREE + 1
    # 特征矩阵、直到MAX_DEGREE阶的切比雪夫多项式列表
    graph = [X] + T_k  # 列表相加
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

# shape为形状元组，不包括batch_size
# 例如shape=(32, )表示预期的输入将是一批32维的向量
X_in = Input(shape=(X.shape[1],))    # 输入的每个feature的维度为1433（feature的列数）
print("X_in.shape:", X_in.shape)   # X_in的Layer名是input_2


# 定义模型架构
# 注意：我们将图卷积网络的参数作为张量列表传递
# 更优雅的做法需要重写Layer基类
# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.

H = Dropout(0.5)(X_in)  # Layer名是dropout_1
H = GraphConvolution(32, support, activation='relu', kernel_regularizer=l2(5e-4))([H] + G)
# Layer名是graph_convolution_1
H = Dropout(0.5)(H)   # Layer名是dropout_2
H = GraphConvolution(256, support, activation=None)([H] + G)   # y.shape[1]
# Layer名是graph_convolution_2
Y = GraphConvolution(64, support, activation=None)([H] + G)   # Layer名是graph_convolution_3

# 编译模型
# Compile model

model = Model(inputs=[X_in] + G, outputs=Y)

# model.load_weights("model_weights.h5")
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# 对中间隐藏层输出的尝试——2021.4.7——————————————————————————————————————
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('graph_convolution_3').output)
graph_convolution_3 = intermediate_layer_model.predict(graph, batch_size=A.shape[0])


# 预测模型在整个数据集上的输出
# Predict on full dataset

# preds = model.predict(graph, batch_size=A.shape[0])

# 将生成的向量写入文件

# 以下是保存96维向量的代码

# preds_vec_file = open("preds_vec-v14.txt", "w", encoding="utf-8")
# preds = list(graph_convolution_2)
# print("len:", len(preds))   # 72380
# print("len1:", len(preds[0]))   # 96
# print("preds[0].type:", type(preds[0]))   # numpy.ndarray
# for p in range(len(preds)):
#     preds[p] = list(preds[p])
#     for q in range(len(preds[p])):
#         preds[p][q] = round(preds[p][q], 6)
# for o in range(len(preds)):
#     # preds_vec_file.write(str(preds[o]) + "\n")
#     if o != len(preds) - 1:
#         preds_vec_file.write(str(preds[o]) + "\n")
#     else:
#         preds_vec_file.write(str(preds[o]))

# 保存进一遍模型即输出的句向量——2021.4.21
preds_vec_file = open("preds_vec-no-pretrain_64.txt", "w", encoding="utf-8")
preds = list(graph_convolution_3)
# print("len:", len(preds))
# print("len1:", len(preds[0]))
# print("preds[0].type:", type(preds[0]))
for i1 in range(len(preds)):
    preds[i1] = list(preds[i1])
for i in range(len(preds)):
    preds_vec_file.write(str(preds[i]) + "\n")

# 打印向量
print("preds:", graph_convolution_3)
print("preds.type:", type(graph_convolution_3))
print("preds.shape:", graph_convolution_3.shape)
# print("preds.shape[0]:", preds.shape[0])
# print("preds.shape[1]:", preds.shape[1])

print("Finish!")
