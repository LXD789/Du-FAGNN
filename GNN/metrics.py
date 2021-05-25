import tensorflow as tf


def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking.带掩码的softmax交叉熵损失"""
    # tensorflow交叉熵计算函数输入中的logits都不是softmax或sigmoid的输出，而是softmax或sigmoid函数的输入，
    #   因为它在函数内部进行sigmoid或softmax操作。而且不能在交叉熵函数前进行softmax或sigmoid，会导致计算会出错。
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None):
    #  计算logits和labels之间的交叉熵
    return tf.reduce_mean(loss)
    # tf.reduce_mean()函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值


def accuracy(preds, labels):
    """Accuracy with masking.带掩码的精确度"""
    # 1.比较预测结果与真实标签是否相等。
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    # correct_prediction：正确预测(bool型数组(大小与做比较的两个数组一致))
    # 判断“tf.argmax(preds, 1)”和“tf.argmax(labels, 1)”是否相等。
    # 若相等，则返回True到correct_prediction，否则返回False。即correct_prediction为bool型。
    # tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引。返回的是一个数组。
    # axis=0时比较每一列的元素；axis=1时比较每一行的元素。
    # 此句意为：判断“preds按行取得的每行最大值组成的数组”和“labels按行取得的每行最大值组成的数组”是否相等。

    # 2.将比较后的返回值(bool型数组)转换为float型数组。
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    # cast(x,dtype,name=None)：将x的数据格式转化成dtype数据类型
    # 此句意为：将correct_prediction转换为float32类型，并将新数据赋值给accuracy_all。
    return tf.reduce_mean(accuracy_all)   # 3.计算accuracy_all总体的平均值
