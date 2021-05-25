from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sklearn import metrics
import pickle as pkl

from utils import *
from models import GNN, MLP

# ——————————————————这个文件需要用到具体的数据集——————————————————————————————
# ——————————————————要从终端输入命令————————————————————————————————————
# 终端命令格式：python train.py

# 一、Settings设置——————————————————————————————————————————————
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'weibo', 'Dataset string.')
# flags.DEFINE_string('dataset', 'twitter15', 'Dataset string.')
# flags.DEFINE_string('dataset', 'twitter-v2', 'Dataset string.')

flags.DEFINE_string('model', 'gnn', 'Model string.')
# flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

flags.DEFINE_integer('epochs', 380, 'Number of epochs to train.')
# epochs = 200
# flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')

flags.DEFINE_integer('batch_size', 512, 'Size of batches per epoch.')
# batch_size = 4096

# flags.DEFINE_integer('input_dim', 300, 'Dimension of input.')
flags.DEFINE_integer('input_dim', 288, 'Dimension of input.')
# flags.DEFINE_integer('input_dim', 769, 'Dimension of input.')
# input_dim = 300

flags.DEFINE_integer('hidden', 96, 'Number of units in hidden layer.')  # 32, 64, 96, 128->这个是隐藏层可以设置的单元数
# flags.DEFINE_integer('hidden', 128, 'Number of units in hidden layer.')
# hidden = 96

flags.DEFINE_integer('steps', 2, 'Number of graph layers.')
# flags.DEFINE_integer('steps', 3, 'Number of graph layers.')
# steps = 2

flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
# dropout = 0.5

flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')  # 5e-4->也可以设为5e-4
# weight_decay = 0.

flags.DEFINE_integer('early_stopping', -1, 'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('early_stopping', 1, 'Tolerance for early stopping (# of epochs).')
# early_stopping = -1

# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')  # Not used


# 二、Load data加载数据————————————————————————————————————————
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = \
    load_data(FLAGS.dataset)


# 三、Some preprocessing一些预处理——————————————————————————————————
# 1.加载训练集
print('loading training set')
train_adj, train_mask = preprocess_adj(train_adj)
train_feature = preprocess_features(train_feature)  # 这个地方有错误&*&*&*&*&*&*&*&*&*&*&*&*&*&*&**&*&*&*&*&*&*&*&*&*&*&&*&*&*

# 2.加载验证集
print('loading validation set')
val_adj, val_mask = preprocess_adj(val_adj)
val_feature = preprocess_features(val_feature)

# 3.加载测试集
print('loading test set')
test_adj, test_mask = preprocess_adj(test_adj)
test_feature = preprocess_features(test_feature)

# 四、对所用模型进行匹配—————————————————————————————————————————
if FLAGS.model == 'gnn':
    model_func = GNN
elif FLAGS.model == 'gcn_cheby':  # not used
    # support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GNN
elif FLAGS.model == 'dense':  # not used
    # support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


# 五、Define placeholders定义占位符——————————————————————————————————
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(None, None, None)),
    'features': tf.placeholder(tf.float32, shape=(None, None, FLAGS.input_dim)),
    'mask': tf.placeholder(tf.float32, shape=(None, None, 1)),
    'labels': tf.placeholder(tf.float32, shape=(None, train_y.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# 六、Create model生成模型——————————————————————————————————————
model = model_func(placeholders, input_dim=FLAGS.input_dim, logging=True)

# 七、Initialize session初始化会话——————————————————————————————————
sess = tf.Session()


# 八、Define model evaluation function定义模型评估函数——————————————————————————
def evaluate(features, support, mask, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)

    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels],
                        feed_dict=feed_dict_val)
    # 定义outs_val，接收sess.run()的返回值。

    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]


# 九、Init variables初始化变量——————————————————————————————————————
sess.run(tf.global_variables_initializer())
# 全局变量初始化

cost_val = []     # 代价变量列表
best_acc = 0      # 最好精度
best_epoch = 0    # 最好轮次
best_cost = 0     # 最少代价
test_doc_embeddings = None    # 测试集的文档嵌入
preds = None                  # 预测
labels = None                 # 标签

print('train start...')
# 十、Train model训练模型————————————————————————————————————————
for epoch in range(FLAGS.epochs):
    t = time.time()    # 定义t接收时间戳
        
    # 1.Training step训练步骤——————————————————
    indices = np.arange(0, len(train_y))
    np.random.shuffle(indices)
    
    train_loss, train_acc = 0, 0
    for start in range(0, len(train_y), FLAGS.batch_size):
        end = start + FLAGS.batch_size
        idx = indices[start:end]

        # （1）Construct feed dictionary构造提要字典(将占位符赋值)
        feed_dict = construct_feed_dict(train_feature[idx], train_adj[idx], train_mask[idx],
                                        train_y[idx], placeholders)

        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        train_loss += outs[1]*len(idx)
        train_acc += outs[2]*len(idx)
    train_loss /= len(train_y)
    train_acc /= len(train_y)

    # 2.Validation验证——————————————————————
    val_cost, val_acc, val_duration, _, _, _ = evaluate(val_feature, val_adj, val_mask, val_y, placeholders)
    cost_val.append(val_cost)
    
    # 3.Test测试—————————————————————————
    test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(test_feature, test_adj,
                                                                            test_mask, test_y, placeholders)

    # 此段代码表示：若当前的测试精度高于已知最好精度，则更新最好精度以及最好epoch、最少代价、测试文档嵌入和预测。
    # 一句话：更新最好精度。
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        best_cost = test_cost
        test_doc_embeddings = embeddings
        preds = pred

    # 4.Print results打印结果———————————————————
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_cost),
          "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(test_acc), 
          "time=", "{:.5f}".format(time.time() - t), "best_acc=", "{:.5f}".format(best_acc))

    if epoch > FLAGS.early_stopping > 0 and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# 十一、Best results最好结果——————————————————————————————————————
print('Best epoch:', best_epoch+1)
print("Test set results:", "cost=", "{:.5f}".format(best_cost),
      "accuracy=", "{:.5f}".format(best_acc))

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(labels, preds, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))

'''
# For visualization
doc_vectors = []
for i in range(len(test_doc_embeddings)):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append(str(np.argmax(test_y[i])) + ' ' + doc_vector_str)

doc_embeddings_str = '\n'.join(doc_vectors)
with open('data/' + FLAGS.dataset + '_doc_vectors.txt', 'w'):
    f.write(doc_embeddings_str)
'''
