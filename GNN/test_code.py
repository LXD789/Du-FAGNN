import json
import tensorflow as tf
import numpy as np


# 将eid与每个事件的源推文vec进行映射————2021.4.5
def eid_source_vec_map():
    # 1.读取eid_list.txt
    eid_list_file = open("data_structure/eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_list_file.readlines()
    for i in range(len(eid_list)):
        eid_list[i] = eid_list[i].replace("\n", "")
    #
    # 2.读取repo_num.json
    repo_num_file = open("data_structure/repo_num.json", "r", encoding="utf-8")
    repo_num_dict = json.load(repo_num_file)

    # 3.首先提取source_post的句向量
    preds_vec_file = open("vector/preds_vec-no-pretrain_64.txt", "r", encoding="utf-8")
    preds_vec = preds_vec_file.readlines()

    source_vec_file = open("vector/source_vec-no-pretrain_64.txt", "w", encoding="utf-8")

    source_position = []   # 存放源推文下标的list
    temp = 0   # 当前的源推文下标
    source_position.append(temp)
    for k, v in repo_num_dict.items():
        if v < 5:
            temp += (v + 1)
            source_position.append(temp)
        elif 5 <= v < 10:
            temp += 6
            source_position.append(temp)
        elif 10 <= v < 15:
            temp += 11
            source_position.append(temp)
        else:
            temp += 16
            source_position.append(temp)

    for j in range(len(source_position)):
        source_vec_file.write(preds_vec[source_position[j]])
    source_vec_file.close()
    print("提取事件源推文向量完毕！")

    # 4.将eid与每个事件的源推文vec进行映射
    eid_sentence_dict_file = open("data_structure/eid_sentence_dict-no-pretrain_64.json", "w", encoding="utf-8")
    source_vec_file = open("vector/source_vec-no-pretrain_64.txt", "r", encoding="utf-8")
    source_vecs = source_vec_file.readlines()

    source_vecs1 = []
    for t in range(len(source_vecs)):
        source_vecs[t] = source_vecs[t].replace("[", "")
        source_vecs[t] = source_vecs[t].replace("]\n", "")
        vec = source_vecs[t].split(", ")
        source_vecs1.append(vec)

    # 将句向量修改为合适的样式
    # 以下是对2维句向量的样式修改代码——2021.4.6
    # for a in range(len(source_vecs)):
    #     # 去除头尾的"[","]"
    #     source_vecs[a] = source_vecs[a].replace("[", "")
    #     source_vecs[a] = source_vecs[a].replace("]", "")
    #     if source_vecs[a].count(" ", 0, 2) == 1:   # 若开头有空格：
    #         source_vecs[a] = source_vecs[a].replace(" ", "", 1)
    #     # 计算中间和末尾的空格数
    #     t = source_vecs[a].count(" ", 1, 13)  # 中间的空格数
    #     u = source_vecs[a].count(" ", 13, len(source_vecs[a]))  # 末尾的空格数
    #     if t != 1:  # 若中间有多个空格，则将多余空格删除，只保留一个空格
    #         source_vecs[a] = source_vecs[a].replace(" ", "", t - 1)
    #     for x in range(u):
    #         source_vecs[a] = source_vecs[a].rstrip(" ")

    eid_sentence_dict = {}
    for eid in eid_list:
        eid_sentence_dict[eid] = source_vecs1[eid_list.index(eid)]
    eid_sentence_js = json.dumps(eid_sentence_dict)
    eid_sentence_dict_file.write(eid_sentence_js)
    print("已生成eid_sentence_vec映射字典文件")


# eid_source_vec_map()


# 将bert词向量对应的词与GCN生成的源推文句向量相结合——2021.4.5,2021.4.6
def make_word_sentence_vec_map():
    # 1.打开eid_sentence_vec_dict和eid_cleanword_dict
    eid_sentence_vec_file = open("data_structure/eid_sentence_dict-no-pretrain_64.json", "r", encoding="utf-8")
    eid_sentence_dict = json.load(eid_sentence_vec_file)
    eid_cleanword_file = open("data_structure/eid_cleanword.json", "r", encoding="utf-8")
    eid_cleanword_dict = json.load(eid_cleanword_file)

    # 2.将eid_sentence_dict中的源推文句向量统一存入一个列表中
    sentence_vec_all = []
    for value in eid_sentence_dict.values():
        sentence_vec_all.append(value)

    # 3.将eid_cleanword_dict中的cleanword统一存入一个列表中
    cleanword_all = []
    for value in eid_cleanword_dict.values():
        cleanword_all.append(value)

    # 构造bert词和句向量的映射字典
    bertvec_file = open("vector/wb-bert256-(-2).txt", "r", encoding="utf-8")  # 存储词和bert词向量
    bertvec_list = []  # 存储bert词的list
    for bertvec in bertvec_file.readlines():
        bertvec_list.append(bertvec.split(" ")[0])

    word_sentence_vec_dict = {}
    for a in range(len(bertvec_list)):
        word_sentence_vec_dict[bertvec_list[a]] = [0.0] * 64   # ————此处若修改向量维度，则也要做相应修改
        # " ".join("%s" % elem for elem in vector)
        for b in range(len(cleanword_all)):
            if cleanword_all[b].find(bertvec_list[a]) != -1:
                word_sentence_vec_dict[bertvec_list[a]] = sentence_vec_all[b]
                cleanword_all[b].replace(bertvec_list[a], "")
                break
            else:
                # word_doubtfeat_dict[bertvec_list[a]] = [0.0]
                continue
    print("done!")

    # 生成文件
    word_sentence_vec_file = open("data_structure/word_sentence_vec_dict-no-pretrain_64.json", "w", encoding="utf-8")
    word_sentence_js = json.dumps(word_sentence_vec_dict, ensure_ascii=False)
    word_sentence_vec_file.write(word_sentence_js)
    print("生成word_sentence_vec_dict映射文件")


# make_word_sentence_vec_map()


# 将bert词向量与句向量相结合——2021.4.6
def concate_bertvec_and_sentence_vec():
    word_sentence_vec_file = open("data_structure/word_sentence_vec_dict-no-pretrain_64.json", "r", encoding="utf-8")
    word_sentence_vec_dict = json.load(word_sentence_vec_file)

    bertvec_file = open("vector/wb-bert256-(-2).txt", "r", encoding="utf-8")
    bertvec_all = bertvec_file.readlines()

    new_vec_file = open("vector/wb_bert_sentence_vec-no-pretrain_64.txt", "w", encoding="utf-8")

    for i in range(len(bertvec_all)):
        word = bertvec_all[i].split(" ")[0]
        vector = bertvec_all[i].split(" ")[1:]
        vector[-1] = vector[-1].replace("\n", "")
        for j in range(len(word_sentence_vec_dict[word])):
            vector.append(word_sentence_vec_dict[word][j])
        # vector.append(word_sentence_vec_dict[word].split(" ")[0])
        # vector.append(word_sentence_vec_dict[word].split(" ")[1])
        new_vec_file.write(word + " ")
        new_vec_file.write(" ".join("%s" % elem for elem in vector))
        new_vec_file.write("\n")
    print("生成bert词向量-句向量融合向量")


# concate_bertvec_and_sentence_vec()


# s = "-0.59666 -1.314816"
# t = s.split(" ")
# print("t:", t)


# 将256维bert词向量同256维GCN句向量add————2021.4.9
def add_bert_gcn_vec():
    word_sentence_vec_file = open("data_structure/word_sentence_vec_dict-no-pretrain_256.json", "r", encoding="utf-8")
    word_sentence_vec_dict = json.load(word_sentence_vec_file)

    bertvec_file = open("vector/wb-bert256-(-2).txt", "r", encoding="utf-8")
    bertvec_all = bertvec_file.readlines()

    new_vec_file = open("vector/wb_bert_sentence_vec-no-pretrain_256.txt", "w", encoding="utf-8")

    for i in range(len(bertvec_all)):
        word = bertvec_all[i].split(" ")[0]
        vector = bertvec_all[i].split(" ")[1:]
        vector[-1] = vector[-1].replace("\n", "")
        new_vector = []
        # 将string类型的bert词向量转换为float
        for j in range(len(vector)):
            new_vector.append(float(vector[j]))
        # new_vector1 = np.array(new_vector)
        # 将GCN句向量list中的string类型转换为float
        for key, value in word_sentence_vec_dict.items():
            for k in range(len(value)):
                value[k] = float(value[k])
            # word_sentence_vec_dict[key] = np.array(value)
            word_sentence_vec_dict[key] = value
        # 两种256维向量add
        bert_gcn_vec = []
        for n in range(len(new_vector)):
            bert_gcn_vec.append(new_vector[n] + word_sentence_vec_dict[word][n])

        # bert_gcn_vec = np.add(new_vector1, word_sentence_vec_dict[word])
        # print("type:", type(bert_gcn_vec))
        # print("shape:", bert_gcn_vec.shape)
        # 将新向量写入文件
        # bert_gcn_vec_list = list(bert_gcn_vec)
        for m in range(len(bert_gcn_vec)):
            bert_gcn_vec[m] = round(bert_gcn_vec[m], 6)
            bert_gcn_vec[m] = str(bert_gcn_vec[m])
        new_vec_file.write(word + " ")
        new_vec_file.write(" ".join("%s" % elem for elem in bert_gcn_vec))
        new_vec_file.write("\n")

        # word = bertvec_all[i].split(" ")[0]
        # vector = bertvec_all[i].split(" ")[1:]
        # vector[-1] = vector[-1].replace("\n", "")
        # for j in range(len(word_sentence_vec_dict[word])):
        #     vector.append(word_sentence_vec_dict[word][j])
        # # vector.append(word_sentence_vec_dict[word].split(" ")[0])
        # # vector.append(word_sentence_vec_dict[word].split(" ")[1])
        # new_vec_file.write(word + " ")
        # new_vec_file.write(" ".join("%s" % elem for elem in vector))
        # new_vec_file.write("\n")
    print("生成bert词向量-句向量融合向量")


# add_bert_gcn_vec()


# 重新生成bertword.txt文件
def remake_bertword():
    bertvec_file = open("vector/wb-bert256-(-2).txt", "r", encoding="utf-8")  # 存储词和bert词向量
    bertword_file = open("data_structure/bertword.txt", "w", encoding="utf-8")  # 存储字词
    bertvec_list = []  # 存储字词的list
    for bertvec in bertvec_file.readlines():
        bertvec_list.append(bertvec.split(" ")[0])
    bertword_file.write(str(bertvec_list))
    print("done!")


# remake_bertword()


# 计算label为0和label为1的数据各有多少
def count_num():
    weibo_label_file = open("data/weibo.txt", "r", encoding="utf-8")
    weibo_label = weibo_label_file.readlines()

    train_zero = 0
    train_one = 0
    test_zero = 0
    test_one = 0
    for i in range(len(weibo_label)):
        label = weibo_label[i].split("\t")[2].replace("\n", "")
        train_or_test = weibo_label[i].split("\t")[1]
        if train_or_test == "train" and label == "0":
            train_zero += 1
        elif train_or_test == "train" and label == "1":
            train_one += 1
        elif train_or_test == "test" and label == "0":
            test_zero += 1
        else:
            test_one += 1
    print("train_zero:", train_zero)
    print("train_one:", train_one)
    print("test_zero:", test_zero)
    print("test_one:", test_one)


# count_num()


# 试验代码
# for t in range(len(source_vecs)):
#     source_vecs[t] = source_vecs[t].replace("[", "")
#     source_vecs[t] = source_vecs[t].replace("]\n", "")
#     vec = source_vecs[t].split(", ")
#     source_vecs1.append(vec)

# 试验代码2——2021.4.9
# t1 = tf.constant([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.])
#
# t2 = tf.constant([0.576347, -1.573465])
#
# sess = tf.Session()
# print("t3:", sess.run(tf.add(t1, t4)))

# 试验代码3——2021.4.10
# s = np.array([[1, 2, 3], [4, 5, 6]])
# t = list(s)
# print("t[0].type:", type(t[0]))
# print("t[0].shape:", t[0].shape)
# print("t[1].type:", type(t[1]))

# 试验代码4——查看新生成的聚合向量的最后一个向量是否与原bert向量的最后一个向量相同——2021.4.12
# f1 = open("vector/wb-bert256-(-2).txt", "r", encoding="utf-8")
# bert_all = f1.readlines()
#
# f2 = open("vector/wb_bert_sentence_vec.txt", "r", encoding="utf-8")
# bert_sentence_all = f2.readlines()
#
# for i in range(1, 6):
#     print("bert_all[-" + str(i) + "]:", bert_all[-i])
#     print("bert_sentence_all[-" + str(i) + "]:", bert_sentence_all[-i])

# 试验代码5——2021.4.17——把bert词向量降到200维
# 读取bert_sentence聚合向量
# bert_sentence_vec_file = open("vector/wb_bert_sentence_vec.txt", "r", encoding="utf-8")
# bert_sentence_vec = bert_sentence_vec_file.readlines()
# bert_sentence_vec_file.close()
#
# new_bert_sent_vec_file = open("vector/200d_wb_bert_sentence_vec.txt", "w", encoding="utf-8")
# for i in range(len(bert_sentence_vec)):
#     word = bert_sentence_vec[i].split(" ")[0]
#     vector = bert_sentence_vec[i].split(" ")[1:]
#     new_vec = vector[:200]
#     new_vec[-1] = new_vec[-1] + "\n"
#     new_vec.insert(0, word)
#     new_bert_sent_vec_file.write(" ".join("%s" % elem for elem in new_vec))
# print("生成200维向量")


# 统计每个词向量文件中的向量维度，防止混淆——2021.5.6
# f = open("vector/wb-bert256-(-2).txt", "r", encoding="utf-8")
# v1 = f.readlines()[0]
# vec = v1.split(" ")[:-1]
# print("len(vec):", len(vec))
