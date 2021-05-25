import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm

# ————————————————这个文件要用到具体的数据集—————————————————————————————————
# ————————————————要从终端输入命令——————————————————————————————————————
# 终端命令格式：python build_graph.py ...(数据集名) ...(窗口大小) ...（带权图：True/False）

if len(sys.argv) < 2:
    sys.exit("Use: python build_graph.py <dataset>")

# settings设置
datasets = ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB',
            'SST1', 'SST2', 'twitter15', 'twitter-v2', 'weibo']
# 定义列表类对象datasets(数据集)，其内容为9个数据集名

dataset = sys.argv[1]  # 接收从外部输入的数据集名，并赋值给dataset
if dataset not in datasets:  # 错误提示
    sys.exit("wrong dataset name")

# 下边一行注释是改错误时添加的
# noinspection PyBroadException
window_size = int(sys.argv[2])
# try:
#     window_size = int(sys.argv[2])
# except Exception:
#     window_size = 3
#     print('using default window size = 3')

# 下边一行注释是改错误时添加的
# noinspection PyBroadException
try:
    weighted_graph = bool(sys.argv[3])
except Exception:
    weighted_graph = False
    print('using default unweighted graph')

truncate = False  # whether to truncate long document是否截断长文件
MAX_TRUNC_LEN = 350  # 设置文本最大长度为350

print('loading raw data')  # 打印“加载行数据”

# 一、load pre-trained word embeddings加载预训练的词嵌入——————————————————
# word_embeddings_dim = 300    # 设置词嵌入的维度为300维
# word_embeddings_dim = 256   # ,512
word_embeddings_dim = 288
word_embeddings = {}  # 设置词嵌入字典


with open('vector/wb_bert_sentence_vec-no-pretrain_32.txt', 'r', encoding='utf-8')as tnb:
    # ,wb-bert256-(-2)
    # wb_bert_sentence_vec-32d
    #
    # wb_bert_sentence_vec-no-pretrain_64
    for line in tnb.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float, data[1:]))

# 二、load document list加载文件列表：将“id+训练集/测试集+标签”添加进相应列表——————————————————
doc_name_list = []  # 文件名列表(是后边两个文件列表的总表)
doc_train_list = []  # 训练集文件列表
doc_test_list = []  # 测试集文件列表

with open('data/' + dataset + '.txt', 'r', encoding='utf-8') as f:  # 此处添加了“encoding='utf-8'”——2020.10.18
    for line in f.readlines():

        doc_name_list.append(line.strip())
        temp = line.split("\t")

        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())

# 测试
# print("doc_name_list:", doc_name_list)

# 三、load raw text加载行(hang)文本————————————————————————————
doc_content_list = []  # 文件内容列表

with open('data/corpus/' + dataset + '.clean.txt', 'r', errors='ignore') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())

# 四、map and shuffle(序列)映射(序列映射指通过相应函数对序列进行操作)和随机排序————————————————
train_ids = []  # 训练集数据id列表
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

# 注：train_ids和test_ids列表中存储的均为整型数值

# 下面是对测试集的数据id提取
test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

ids = train_ids + test_ids  # 将训练集数据id列表与测试集数据id列表合并为一个列表，赋值给ids

shuffle_doc_name_list = []  # 随机排序的文件名列表
shuffle_doc_words_list = []  # 随机排序的文件词列表
for i in ids:
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])  # 这里有错误*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*

# 五、build corpus vocabulary建立语料库词汇——————————————————————————————
word_set = set()  # 定义set类对象word_set(词集)
# set()函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    word_set.update(words)

vocab = list(word_set)  # 定义列表类对象vocab，赋值为转换成列表类型的word_set。
vocab_size = len(vocab)  # 定义整型变量vocab_size(词汇量)，赋值为vocab所含元素个数。

word_id_map = {}  # 定义字典类对象word_id_map(词id映射)
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

# 六、initialize out-of-vocabulary word embeddings初始化词汇表外词嵌入———————————————————————
oov = {}  # oov:out-of-vocabulary词汇表外(即不在vocab中的词汇)词汇字典
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)

# 七、build label list建立标签列表————————————————————————————————————————
label_set = set()  # 定义set类对象label_set(标签集)
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

# 八、select 90% training set选择90%训练集—————————————————————————————————————
train_size = len(train_ids)  # 定义整型变量train_size(训练集元素总数)，赋值为train_ids(训练集数据id列表)的元素个数。
# val_size = int(0.1 * train_size)  # 定义整型变量val_size(验证集元素个数)，赋值为转换成整型的0.1与train_size的乘积。
val_size = int(0.11 * train_size)  # 0.053
real_train_size = train_size - val_size  # 定义整型real_train_size(真实训练集元素个数)，赋值为总数与验证集元素数之差。
test_size = len(test_ids)  # 定义整型变量test_size(测试集元素数)，赋值为test_size(测试集数据id列表)的元素个数。

# train_size = len(train_ids)  # 定义整型变量train_size(训练集元素总数)，赋值为train_ids(训练集数据id列表)的元素个数。
# # val_size = int(0.1 * train_size)  # 定义整型变量val_size(验证集元素个数)，赋值为转换成整型的0.1与train_size的乘积。
# real_train_size = train_size  # 定义整型real_train_size(真实训练集元素个数)，赋值为总数与验证集元素数之差。
# test_size = len(test_ids)  # 定义整型变量test_size(测试集元素数)，赋值为test_size(测试集数据id列表)的元素个数。


# define graph function定义图函数————————————————————————————————————————
def build_graph(start, end):
    x_adj = []  # 训练集x邻接矩阵列表
    x_feature = []  # 训练集x特征矩阵列表
    y = []  # 标签y列表
    doc_len_list = []  # 文档长度列表
    vocab_set = set()  # 词汇集

    for i in tqdm(range(start, end)):
        doc_words = shuffle_doc_words_list[i].split()
        # 定义列表doc_words(文档词)，赋值为shuffle_doc_words_list(随机排序的文件名列表)的当前元素分割后的列表。
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}  # 定义字典类对象doc_word_id_map(文档词id映射)
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows滑动窗口——————————————————————————
        windows = []  # 定义列表类对象windows(窗口)
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            # (此处的else实现了“窗口滑动”和“向窗口列表中添加元素”)———————
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}  # 定义字典类对象word_pair_count(词对计数)
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]  # 定义string对象word_p，赋值为windows列表当前元素window列表下标为p的元素
                    word_p_id = word_id_map[word_p]  # 定义整型word_p_id，赋值为word_id_map字典中键word_p对应的值
                    word_q = window[q]  # 定义string对象word_q，赋值为windows列表当前元素window列表下标为q的元素
                    word_q_id = word_id_map[word_q]  # 定义整型word_q_id，赋值为word_id_map字典中键word_q对应的值
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)  # 定义元组word_pair_key，元素为word_p_id, word_q_id

                    # word co-occurrences as weights单词共现作为权重——————————————
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

                    # bi-direction双向————————————————————————————
                    # 此部分意为将word_p_id与word_q_id位置颠倒，以严密整个模型的逻辑。
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []  # 行列表
        col = []  # 列 列表
        weight = []  # 权重列表
        features = []  # 特征列表

        for key in word_pair_count:
            p = key[0]  # 将key的第一个元素赋值给整型变量p
            q = key[1]  # 将key的第二个元素赋给整型变量q
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        # 定义矩阵adj，赋值为压缩稀疏行矩阵(按行压缩)

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):

            features.append(word_embeddings[k] if k in word_embeddings else oov[k])

        x_adj.append(adj)
        x_feature.append(features)

    # one-hot labels独热标签——————————————————————————
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)

    # 此部分：将独热编码列表中下标为“标签在label_list中的索引”的那一位元素标记为1。将独热编码列表添加进y列表，
    #         并将y列表转换为数组。

    return x_adj, x_feature, y, doc_len_list, vocab_set


# 九、build graph建立图(调用上边的build_graph)————————————————————————————————————
print('building graphs for training')
x_adj, x_feature, y, _, _ = build_graph(start=0, end=real_train_size)
# 有时候单个独立下划线是用作一个名字，来表示某个变量是临时的或无关紧要的。
print('building graphs for training + validation')
allx_adj, allx_feature, ally, doc_len_list_train, vocab_train = build_graph(start=0, end=train_size)
print('building graphs for test')
tx_adj, tx_feature, ty, doc_len_list_test, vocab_test = build_graph(start=train_size, end=train_size + test_size)
doc_len_list = doc_len_list_train + doc_len_list_test

# 十、statistics统计—————————————————————————————————————————————————
print('max_doc_length', max(doc_len_list), 'min_doc_length', min(doc_len_list),
      'average {:.2f}'.format(np.mean(doc_len_list)))
print('training_vocab', len(vocab_train), 'test_vocab', len(vocab_test),
      'intersection', len(vocab_train & vocab_test))

# 十一、dump objects转储对象(存储对象)————————————————————————————————————————
with open("data/ind.{}.x_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)

with open("data/ind.{}.x_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)

with open("data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("data/ind.{}.tx_adj".format(dataset), 'wb') as f:
    pkl.dump(tx_adj, f)

with open("data/ind.{}.tx_embed".format(dataset), 'wb') as f:
    pkl.dump(tx_feature, f)

with open("data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("data/ind.{}.allx_adj".format(dataset), 'wb') as f:
    pkl.dump(allx_adj, f)

with open("data/ind.{}.allx_embed".format(dataset), 'wb') as f:
    pkl.dump(allx_feature, f)

with open("data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)
