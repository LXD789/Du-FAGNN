import matplotlib

# encoding: utf-8
import matplotlib.pyplot as plt
import numpy as np


def zhexiantu():
    # 折线图
    x = [0, 0.025, 0.1, 0.15, 0.2, 0.24, 0.29, 0.35, 0.4, 0.5, 0.6, 0.75, 0.95, 1.25, 1.75, 1.875, 2]  # 点的横坐标
    k1 = [0.44, 0.49, 0.56, 0.62, 0.67, 0.71, 0.74, 0.76, 0.77, 0.78, 0.786, 0.791, 0.795, 0.798, 0.8, 0.801, 0.803]
    # SVM-RBF的纵坐标
    k2 = [0.53, 0.59, 0.66, 0.72, 0.77, 0.79, 0.801, 0.807, 0.812, 0.814, 0.816, 0.819, 0.822, 0.823, 0.825, 0.827,
          0.83]
    # TextGCN的纵坐标
    k3 = [0.387, 0.581, 0.624, 0.645, 0.677, 0.731, 0.742, 0.753, 0.774, 0.785, 0.796, 0.828, 0.839, 0.841, 0.843,
          0.844,
          0.846]
    # TextING
    # k4 = [0.68, 0.734, 0.785, 0.843, 0.856, 0.863, 0.868, 0.87, 0.874, 0.879, 0.886, 0.89, 0.892, 0.895, 0.897, 0.903,
    #       0.907]
    k4 = [0.6, 0.63, 0.67, 0.7, 0.733, 0.765, 0.78, 0.81, 0.842, 0.86, 0.87, 0.885, 0.892, 0.895, 0.897, 0.903,
          0.907]
    # RvNN
    k5 = [0.884, 0.886, 0.889, 0.89, 0.893, 0.897, 0.899, 0.9, 0.903, 0.908, 0.911, 0.915, 0.916, 0.916, 0.916, 0.916,
          0.916]
    # PPC_RNN+CNN
    k6 = [0.5, 0.677, 0.699, 0.753, 0.774, 0.796, 0.806, 0.849, 0.871, 0.882, 0.903, 0.916, 0.92, 0.924, 0.928, 0.935,
          0.946]
    # Bi-FAGNN
    plt.plot(x, k1, 's-', color='g', label="SVM-RBF")  # s-:方形
    plt.plot(x, k2, 'o-', color='#FF8000', label="TextGCN")  # o-:圆形
    plt.plot(x, k3, '^-', color='#0000CD', label='TextING')
    plt.plot(x, k4, 'D-', color='#DEB887', label='RvNN')  # #A0522D
    plt.plot(x, k5, 'x-', color='#9400D3', label='PPC_RNN+CNN')  # #9370DB
    plt.plot(x, k6, 'd-', color='r', label='Du-FAGNN')
    plt.xlabel("Detection Deadline(Hours)", fontsize=12)
    plt.ylabel("Acc.", fontsize=13)
    plt.legend(loc="best")  # 图例
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
    plt.show()


# zhexiantu()


def zhuzhuangtu1():
    # encoding: utf-8
    # 柱状图
    # TextING = 0.842
    # TextGCN = 0.837
    # BiFAGNN = 0.9
    model_names = ['TextING', 'TextGCN', 'Du-FAGNN']
    y = [0.842, 0.837, 0.9]

    # model_names2 = ['TextING', 'Bi-FAGNN']
    # y2 = [0.858, 0.936]

    # x = range(len(model_names))

    # x = np.arange(2)  # 总共有几组，就设置成几，我们这里有三组，所以设置为3
    # plt.bar(x, y, width=0.5, color=['#FF8000', 'y', '#008B8B'])
    plt.bar('TextING', 0.842, width=0.25, color='#FF8000', label='TextING')
    plt.bar('TextGCN', 0.837, width=0.25, color='y', label='TextGNN')
    plt.bar('Du-FAGNN', 0.9, width=0.25, color='#008B8B', label='Du-FAGNN')

    # total_width, n = 0.3, 3  # n有多少个类型
    # width = total_width / n
    # x = x - (total_width - width) / 2
    # plt.bar(x, TextING, color="#FF8000", width=0.02, label='a1 ')
    # plt.bar(x + 0.05, TextGCN, color="y", width=0.02, label='a2')
    # plt.bar(x + 2 * 0.05, BiFAGNN, color="#008B8B", width=0.02, label='a3')
    # plt.xlabel("Detection Deadline(Hours)", fontsize=12)
    plt.ylabel("Acc.", fontsize=17)

    # 给柱子顶上加数字
    plt.text('TextING', y[0] + 0.001, '%.3f' % (y[0] + 0.001), ha='center', va='bottom', fontsize=15)
    plt.text('TextGCN', y[1] + 0.001, '%.3f' % (y[1] + 0.001), ha='center', va='bottom', fontsize=15)
    plt.text('Du-FAGNN', y[2] + 0.001, '%.3f' % (y[2] + 0.001), ha='center', va='bottom', fontsize=15)

    plt.legend(loc="best", fontsize=15)
    plt.ylim((0.8, 0.92))
    my_y_ticks = np.arange(0.8, 0.93, 0.05)
    plt.xticks(['TextING', 'TextGCN', 'Du-FAGNN'], fontsize=15)
    plt.yticks(my_y_ticks, fontsize=15)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
    plt.show()


# zhuzhuangtu1()


def zhuzhuangtu2():
    # encoding: utf-8
    # 柱状图
    # TextING = 0.842
    # TextGCN = 0.837
    # BiFAGNN = 0.9
    # model_names = ['TextING', 'TextGCN', 'Bi-FAGNN']
    # y = [0.842, 0.837, 0.9]

    model_names = ['TextING', 'Du-FAGNN']
    y = [0.858, 0.936]

    x = range(len(model_names))

    # x = np.arange(2)  # 总共有几组，就设置成几，我们这里有三组，所以设置为3
    plt.bar('TextING', 0.858, width=0.2, color='#FF8000', label='TextING')
    plt.bar('Du-FAGNN', 0.936, width=0.2, color='y', label='Du-FAGNN')
    # plt.bar(x + 0.5, width=0.2, color=['r', 'b'])
    # plt.xticks(x, model_names)

    # total_width, n = 0.3, 3  # n有多少个类型
    # width = total_width / n
    # x = x - (total_width - width) / 2
    # plt.bar(x, TextING, color="#FF8000", width=0.02, label='a1 ')
    # plt.bar(x + 0.05, TextGCN, color="y", width=0.02, label='a2')
    # plt.bar(x + 2 * 0.05, BiFAGNN, color="#008B8B", width=0.02, label='a3')
    # plt.xlabel("Detection Deadline(Hours)", fontsize=12)
    plt.ylabel("Acc.", fontsize=17)

    # 柱子顶部加数字
    plt.text('TextING', y[0] + 0.001, '%.3f' % (y[0] + 0.001), ha='center', va='bottom', fontsize=15)
    plt.text('Du-FAGNN', y[1] + 0.001, '%.3f' % (y[1] + 0.001), ha='center', va='bottom', fontsize=15)
    # for a, b in zip(x, y):
    #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)

    plt.legend(loc="best", fontsize=15)
    # plt.xticks(range(2), ['TextING', 'TextGCN', 'Bi-FAGNN'])
    plt.ylim((0.8, 0.95))
    my_y_ticks = np.arange(0.8, 0.95, 0.05)
    plt.xticks(['TextING', 'Du-FAGNN'], fontsize=15)
    plt.yticks(my_y_ticks, fontsize=15)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
    plt.show()


# zhuzhuangtu2()


