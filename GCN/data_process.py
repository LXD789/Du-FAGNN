import json
import sys
import numpy as np
import scipy.sparse as sp
from utils import *


def make_content_file():  # 生成.content文件
    # weibo_info_file = open("Weibo.txt", "r", encoding="utf-8")
    # weibo_info = weibo_info_file.readlines()   # Weibo.txt文件内容

    eid_file = open("eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_file.readlines()   # eid_list

    eid_repoid_file = open("eid_postid_dict.json", "r", encoding="utf-8")
    eid_postid_dict = json.load(eid_repoid_file)   # eid_postid_dict

    eid_label_file = open("eid_label_dict.json", "r", encoding="utf-8")
    eid_label_dict = json.load(eid_label_file)   # eid_label_dict

    content_file = open("weibo.content", "w", encoding="utf-8")   # .content文件

    for eid in eid_list:
        print("eid:", eid)
        f = open("weiboset1vec/" + eid.replace("\n", "") + "_vec.txt", "r", encoding="utf-8")
        lists = f.readlines()   # 某个事件的全部bert-feat
        f1 = open("record/" + eid.replace("\n", "") + "_check.txt", "w", encoding="utf-8")
        if len(lists) < 6:   # 若当前事件中的repost个数小于5，则将全部的句向量导入content文件
            for feat_string in lists:
                feat = []  # 某条推文的bert-feat
                feat.extend(feat_string.split(", "))
                feat[0] = feat[0].lstrip("[")
                feat[-1] = feat[-1].rstrip("]\n")
                # for i in range(len(feat)):
                #     feat[i] = float(feat[i])
                if lists.index(feat_string) == 0:  # 若当前为事件源
                    print("lists.index(feat_string):", lists.index(feat_string))
                    feat.insert(0, eid.replace("\n", ""))  # 将eid添加为feat列表的第一个元素
                    feat.append(eid_label_dict[eid.replace("\n", "")])  # 将label添加为feat列表的最后一个元素
                else:  # 若当前为响应推文
                    print("lists.index(feat_string):", lists.index(feat_string))
                    feat.insert(0, eid_postid_dict[eid.replace("\n", "")][lists.index(feat_string) - 1])
                    # 将repoid添加为feat列表的第一个元素
                    feat.append(eid_label_dict[eid.replace("\n", "")])
                    # feat.insert(-1, eid_label_dict[eid.replace("\n", "")])
                    # 将eid的label作为当前响应推文的label，添加为feat列表的最后一个元素

                # 将feat列表写入.content文件
                if lists.index(feat_string) != (len(lists) - 1):
                    f1.write("\t".join("%s" % elem for elem in feat))
                    f1.write("\n")
                else:
                    f1.write("\t".join("%s" % elem for elem in feat))
                # if lists.index(feat_string) != (len(lists) - 1):
                #     content_file.write("\t".join("%s" % elem for elem in feat))
                #     content_file.write("\n")
                # else:
                #     content_file.write("\t".join("%s" % elem for elem in feat))
        elif 6 <= len(lists) < 11:   # 若当前事件repost个数大于等于5且小于10，则将“事件源+repost”共6个向量导入content文件
            for i in range(6):
                feat = []  # 某条推文的bert-feat
                feat.extend(lists[i].split(", "))
                feat[0] = feat[0].lstrip("[")
                feat[-1] = feat[-1].rstrip("]\n")
                if i == 0:  # 若当前为事件源
                    print("lists.index(feat_string):", i)
                    feat.insert(0, eid.replace("\n", ""))  # 将eid添加为feat列表的第一个元素
                    feat.append(eid_label_dict[eid.replace("\n", "")])  # 将label添加为feat列表的最后一个元素
                else:  # 若当前为响应推文
                    print("lists.index(feat_string):", i)
                    feat.insert(0, eid_postid_dict[eid.replace("\n", "")][i - 1])
                    # 将repoid添加为feat列表的第一个元素
                    feat.append(eid_label_dict[eid.replace("\n", "")])
                    # feat.insert(-1, eid_label_dict[eid.replace("\n", "")])
                    # 将eid的label作为当前响应推文的label，添加为feat列表的最后一个元素

                # 将feat列表写入.content文件
                if i != (len(lists) - 1):
                    f1.write("\t".join("%s" % elem for elem in feat))
                    f1.write("\n")
                else:
                    f1.write("\t".join("%s" % elem for elem in feat))
                # if i != (len(lists) - 1):
                #     content_file.write("\t".join("%s" % elem for elem in feat))
                #     content_file.write("\n")
                # else:
                #     content_file.write("\t".join("%s" % elem for elem in feat))

        elif 11 <= len(lists) < 16:
            for j in range(11):
                feat = []  # 某条推文的bert-feat
                feat.extend(lists[j].split(", "))
                feat[0] = feat[0].lstrip("[")
                feat[-1] = feat[-1].rstrip("]\n")
                if j == 0:  # 若当前为事件源
                    print("lists.index(feat_string):", j)
                    feat.insert(0, eid.replace("\n", ""))  # 将eid添加为feat列表的第一个元素
                    feat.append(eid_label_dict[eid.replace("\n", "")])  # 将label添加为feat列表的最后一个元素
                else:  # 若当前为响应推文
                    print("lists.index(feat_string):", j)
                    feat.insert(0, eid_postid_dict[eid.replace("\n", "")][j - 1])
                    # 将repoid添加为feat列表的第一个元素
                    feat.append(eid_label_dict[eid.replace("\n", "")])
                    # feat.insert(-1, eid_label_dict[eid.replace("\n", "")])
                    # 将eid的label作为当前响应推文的label，添加为feat列表的最后一个元素

                # 将feat列表写入.content文件
                if j != (len(lists) - 1):
                    f1.write("\t".join("%s" % elem for elem in feat))
                    f1.write("\n")
                else:
                    f1.write("\t".join("%s" % elem for elem in feat))
                # if j != (len(lists) - 1):
                #     content_file.write("\t".join("%s" % elem for elem in feat))
                #     content_file.write("\n")
                # else:
                #     content_file.write("\t".join("%s" % elem for elem in feat))
        else:
            for k in range(16):
                feat = []  # 某条推文的bert-feat
                feat.extend(lists[k].split(", "))
                feat[0] = feat[0].lstrip("[")
                feat[-1] = feat[-1].rstrip("]\n")
                if k == 0:  # 若当前为事件源
                    print("lists.index(feat_string):", k)
                    feat.insert(0, eid.replace("\n", ""))  # 将eid添加为feat列表的第一个元素
                    feat.append(eid_label_dict[eid.replace("\n", "")])  # 将label添加为feat列表的最后一个元素
                else:  # 若当前为响应推文
                    print("lists.index(feat_string):", k)
                    feat.insert(0, eid_postid_dict[eid.replace("\n", "")][k - 1])
                    # 将repoid添加为feat列表的第一个元素
                    feat.append(eid_label_dict[eid.replace("\n", "")])
                    # feat.insert(-1, eid_label_dict[eid.replace("\n", "")])
                    # 将eid的label作为当前响应推文的label，添加为feat列表的最后一个元素

                # 将feat列表写入.content文件
                if k != (len(lists) - 1):
                    f1.write("\t".join("%s" % elem for elem in feat))
                    f1.write("\n")
                else:
                    f1.write("\t".join("%s" % elem for elem in feat))
                # if k != (len(lists) - 1):
                #     content_file.write("\t".join("%s" % elem for elem in feat))
                #     content_file.write("\n")
                # else:
                #     content_file.write("\t".join("%s" % elem for elem in feat))

    print("Make content file finished!")
    # 试验区：
    # f = open("weiboset1vec/4010312877_vec.txt", "r", encoding="utf-8")
    # lists = f.readlines()
    # feat0 = lists[0]
    # feat = []
    # feat.extend(feat0.split(", "))
    # feat[0] = feat[0].lstrip("[")
    # feat[-1] = feat[-1].rstrip("]\n")
    # # for i in range(len(feat)):
    # #     feat[i] = float(feat[i])
    # print("feat:", feat)


# make_content_file()


# 2021.4.2
def fine():

    eid_label_file = open("eid_label_dict.json", "r", encoding="utf-8")
    eid_label_dict = json.load(eid_label_file)  # eid_label_dict

    f = open("weiboset1vec/3586313934109778_vec.txt", "r", encoding="utf-8")
    lists = f.readlines()  # 某个事件的全部bert-feat
    f1 = open("record/3586313934109778_check.txt", "w", encoding="utf-8")
    if len(lists) < 6:  # 若当前事件中的repost个数小于5，则将全部的句向量导入content文件
        for feat_string in lists:
            feat = []  # 某条推文的bert-feat
            feat.extend(feat_string.split(", "))
            feat[0] = feat[0].lstrip("[")
            feat[-1] = feat[-1].rstrip("]\n")
            # for i in range(len(feat)):
            #     feat[i] = float(feat[i])
            if lists.index(feat_string) == 0:  # 若当前为事件源
                print("lists.index(feat_string):", lists.index(feat_string))
                feat.insert(0, "3586313934109778")  # 将eid添加为feat列表的第一个元素
                feat.append(eid_label_dict["3586313934109778"])  # 将label添加为feat列表的最后一个元素
            # else:  # 若当前为响应推文
            #     print("lists.index(feat_string):", lists.index(feat_string))
            #     feat.insert(0, eid_postid_dict[eid.replace("\n", "")][lists.index(feat_string) - 1])
            #     # 将repoid添加为feat列表的第一个元素
            #     feat.append(eid_label_dict[eid.replace("\n", "")])
            #     # feat.insert(-1, eid_label_dict[eid.replace("\n", "")])
            #     # 将eid的label作为当前响应推文的label，添加为feat列表的最后一个元素

            # 将feat列表写入.content文件
            if lists.index(feat_string) != (len(lists) - 1):
                f1.write("\t".join("%s" % elem for elem in feat))
                f1.write("\n")
            else:
                f1.write("\t".join("%s" % elem for elem in feat))
            # if lists.index(feat_string) != (len(lists) - 1):
            #     content_file.write("\t".join("%s" % elem for elem in feat))
            #     content_file.write("\n")
            # else:
            #     content_file.write("\t".join("%s" % elem for elem in feat))


# fine()


def fine2():   # 2021.4.4重写make_content_file
    eid_file = open("eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_file.readlines()  # eid_list

    for i in range(len(eid_list)):
        eid_list[i] = eid_list[i].replace("\n", "")

    content_file = open("weibo.content", "w", encoding="utf-8")  # .content文件
    for eid in eid_list:
        f = open("record/" + eid + "_check.txt", "r", encoding="utf-8")
        weibo_all = f.readlines()
        if "\n" in weibo_all[-1]:
            for j in range(len(weibo_all)):
                content_file.write(weibo_all[j])
        else:
            for k in range(len(weibo_all)):
                content_file.write(weibo_all[k])
            content_file.write("\n")
    print("Make new content file done!")


# fine2()

# 试验区：
# f = open("weiboset1vec/4010312877_vec.txt", "r", encoding="utf-8")
#
# eid_repoid_file = open("eid_postid_dict.json", "r", encoding="utf-8")
# eid_postid_dict = json.load(eid_repoid_file)   # eid_postid_dict
#
# eid_label_file = open("eid_label_dict.json", "r", encoding="utf-8")
# eid_label_dict = json.load(eid_label_file)   # eid_label_dict
#
# content_file = open("4010312877.content", "w", encoding="utf-8")   # .content文件
# lists = f.readlines()
# for i in range(16):
#     feat = []  # 某条推文的bert-feat
#     feat.extend(lists[i].split(", "))
#     feat[0] = feat[0].lstrip("[")
#     feat[-1] = feat[-1].rstrip("]\n")
#     print("feat:", feat)
#     if i == 0:  # 若当前为事件源
#         print("lists.index(feat_string):", i)
#         feat.insert(0, "4010312877")  # 将eid添加为feat列表的第一个元素
#         feat.append(eid_label_dict["4010312877"])  # 将label添加为feat列表的最后一个元素
#         print("feat1:", feat)
#     else:  # 若当前为响应推文
#         print("lists.index(feat_string):", i)
#         feat.insert(0, eid_postid_dict["4010312877"][i - 1])
#         # 将repoid添加为feat列表的第一个元素
#         feat.append(eid_label_dict["4010312877"])
#         print("feat1:", feat)
#         # feat.insert(-1, eid_label_dict[eid.replace("\n", "")])
#         # 将eid的label作为当前响应推文的label，添加为feat列表的最后一个元素
#
#         # 将feat列表写入.content文件
#     if i != (len(lists) - 1):
#         content_file.write("\t".join("%s" % elem for elem in feat))
#         content_file.write("\n")
#     else:
#         content_file.write("\t".join("%s" % elem for elem in feat))

# feat0 = lists[0]
# feat = []
# feat.extend(feat0.split(", "))
# feat[0] = feat[0].lstrip("[")
# feat[-1] = feat[-1].rstrip("]\n")
# for i in range(len(feat)):
#     feat[i] = float(feat[i])
# print("feat:", feat)


# with open("weiboset1vec/3911910070304468_vec.txt", "r", encoding="utf-8")as f:
#     print("len1:", len(f.readlines()))

# with open("3909231033736573.txt", "r", encoding="utf-8")as f1:
#     print("len2:", len(f1.readlines()))
# 
# with open("eid_postid_dict.json", "r", encoding="utf-8")as f3:
#     dict1 = json.load(f3)
#     print("len3:", len(dict1["3909231033736573"]))

def make_response_file():   # 尝试生成.cite文件——————2021.3.30
    repo_num_file = open("repo_num.json", "r", encoding="utf-8")
    repo_num_dict = json.load(repo_num_file)

    eid_postid_file = open("eid_postid_dict.json", "r", encoding="utf-8")
    eid_postid_dict = json.load(eid_postid_file)

    weibo_response_file = open("weibo.cites", "w", encoding="utf-8")

    for k, v in repo_num_dict.items():

        if v < 5:
            for i in range(v):
                weibo_response_file.write(k + "\t" + eid_postid_dict[k][i] + "\n")
        elif 5 <= v < 10:
            for j in range(5):
                weibo_response_file.write(k + "\t" + eid_postid_dict[k][j] + "\n")
        elif 10 <= v < 15:
            for m in range(10):
                weibo_response_file.write(k + "\t" + eid_postid_dict[k][m] + "\n")
        else:
            for n in range(15):
                weibo_response_file.write(k + "\t" + eid_postid_dict[k][n] + "\n")

    print("Make response file done!")


# make_response_file()


# 打印.content文件查看错误出在哪里————2021.3.30
def check_bug():
    content_file = open("data/weibo/weibo.content", "r", encoding="utf-8")
    new_content_file = open("new_weibo.content", "w", encoding="utf-8")
    weibo_content = content_file.readlines()
    temp_content = open("temp.content", "w", encoding="utf-8")

    list1 = weibo_content[0].split("\t")
    temp1 = list1[-1].replace("\n", "")   # temp1=特征的最后一位数
    temp2 = list1[-2]   # temp2=label
    print("temp1:", temp1)
    print("temp2:", temp2)

    list1[-1] = temp2 + "\n"
    list1[-2] = temp1
    weibo_content[0] = "\t".join("%s" % elem for elem in list1)

    for i in range(len(weibo_content)):
        new_content_file.write(weibo_content[i])

    new_content_file.close()

    new_cf = open("new_weibo.content", "r", encoding="utf-8")
    new_cf1 = new_cf.readlines()

    for i in range(10):
        temp_content.write(new_cf1[i])

    cites_file = open("data/weibo/weibo.cites", "r", encoding="utf-8")
    cites = cites_file.readlines()
    temp_cites_file = open("temp.cites", "w", encoding="utf-8")
    for j in range(9):
        temp_cites_file.write(cites[j])


# check_bug()


def check_bug1():
    weibo_content_file = open("data/weibo/weibo.content", "r", encoding="utf-8")
    weibo_content = weibo_content_file.readlines()
    error_list = []
    error_dict = {}
    for i in range(len(weibo_content)):
        wid = weibo_content[i].split("\t")[0]
        label = weibo_content[i].split("\t")[-1].replace("\n", "")
        if label != "0" and label != "1":
            error_list.append(wid)
            error_dict[i] = wid
    # print("error_list:", error_list)
    # print("error_dict:", error_dict)

    for k, v in error_dict.items():
        list1 = weibo_content[k].split("\t")
        temp1 = list1[-1].replace("\n", "")
        temp2 = list1[-2]
        print("temp1:", temp1)
        print("temp2:", temp2)
        list1[-1] = temp2 + "\n"
        list1[-2] = temp1
        weibo_content[k] = "\t".join("%s" % elem for elem in list1)

    mid_weibo_file = open("mid_weibo.content", "w", encoding="utf-8")
    for j in range(len(weibo_content)):
        mid_weibo_file.write(weibo_content[j])


# check_bug1()


def check_larger():
    weibo_content_file = open("data/weibo/weibo.content", "r", encoding="utf-8")
    weibo_content = weibo_content_file.readlines()
    lager_list = []
    lager_dict = {}
    for i in range(len(weibo_content)):
        wid = weibo_content[i].split("\t")[0]
        feat = weibo_content[i].split("\t")[1:-1]
        if len(feat) != 256:
            lager_list.append(wid)
            lager_dict[i] = wid

    for k, v in lager_dict.items():
        list1 = weibo_content[k].split("\t")
        list2 = []
        wid = list1[0]
        label = list1[-1]
        t_feat = list1[1:257]
        list2.append(wid)
        for j in range(len(t_feat)):
            list2.append(t_feat[j])
        list2.append(label)
        weibo_content[k] = "\t".join("%s" % elem for elem in list2)

    end_weibo_file = open("new_weibo.content", "w", encoding="utf-8")
    for j in range(len(weibo_content)):
        end_weibo_file.write(weibo_content[j])


# check_larger()


def final_check():
    end_weibo_file = open("data/weibo/weibo.content", "r", encoding="utf-8")
    end_weibo = end_weibo_file.readlines()
    test_file = open("test.content", "w", encoding="utf-8")

    for i in range(1056):
        test_file.write(end_weibo[i])

    weibo_cites_file = open("data/weibo/weibo.cites", "r", encoding="utf-8")
    weibo_cites = weibo_cites_file.readlines()
    test_file1 = open("test.cites", "w", encoding="utf-8")

    for j in range(990):
        test_file1.write(weibo_cites[j])


# final_check()


def check_again():
    weibo_content_file = open("data/weibo/weibo.content", "r", encoding="utf-8")
    weibo_content = weibo_content_file.readlines()
    list3 = []
    wid = weibo_content[70].split("\t")[0]
    feat = weibo_content[70].split("\t")[1:-1]
    label = weibo_content[70].split("\t")[-1]

    feat[0] = feat[0].replace("[", "")

    list3.append(wid)
    list3.append(feat)
    list3.append(label)
    weibo_content[70] = "\t".join("%s" % elem for elem in list3)
    weibo_content_file.close()

    weibo_content_file1 = open("data/weibo/weibo.content", "w", encoding="utf-8")
    for k in range(len(weibo_content)):
        weibo_content_file1.write(weibo_content[k])
    weibo_content_file1.close()

    # weibo_content_file2 = open("data/weibo/weibo.content", "r", encoding="utf-8")
    # weibo_content1 = weibo_content_file2.readlines()
    # print("weibo_content[70]:", weibo_content1[70])
    # t_file = open("t70.content", "w", encoding="utf-8")
    # t_file.write(weibo_content1[70])

    # check_list = []
    # check_dict = {}
    # for i in range(len(weibo_content)):
    #     wid = weibo_content[i].split("\t")[0]
    #     f = weibo_content[i]
    #     if f.find("['-0.0071816',") != -1:
    #         check_list.append(wid)
    #         check_dict[i] = wid

    # print("check_list:", check_list)
    # print("check_dict:", check_dict)   # {70: '13394555852'}


# check_again()


def load_data(path="", dataset="test"):
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
    print("feature行数：", features.shape[0])   # ----->10
    print("feature列数：", features.shape[1])   # ----->256
    #
    # 提取样本的标签，并将其转换为one-hot编码形式
    labels = encode_onehot(idx_features_labels[:, -1])
    print("label行数：", labels.shape[0])
    print("label列数：", labels.shape[1])
    # label行数： 10
    # label列数： 1
    #
    # build graph
    # 样本的id数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int64)
    print("idx:", idx)
    print("idx.type:", type(idx))
    print("idx.shape:", idx.shape)
    # 由样本id到样本索引的映射字典
    idx_map = {j: i for i, j in enumerate(idx)}  # i为索引，j为样本id。所以该字典为：{样本id:样本索引}
    print("idx_map:", idx_map)
    # 样本之间的引用关系数组
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int64)
    print("edges_unordered行数：", edges_unordered.shape[0])
    print("edges_unordered列数：", edges_unordered.shape[1])
    print("edges_unordered[:, 0]:", edges_unordered[:, 0])
    print("edges_unordered[:, 1]:", edges_unordered[:, 1])
    # edges_unordered行数： 9
    # edges_unordereda列数： 2
    #
    # 将样本之间的引用关系用样本索引之间的关系表示——————————>即：将样本id换成其对应的样本索引
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    # print("edges.type:", type(edges))  # ndarray
    print("edges行数：", edges.shape[0])
    print("edges列数：", edges.shape[1])
    # edges行数： 9
    # edges列数： 2
    #
    # edges[:, 0] [0 0 0 0 0 0 0 0 0]
    # edges[:, 1] [1 2 3 4 5 6 7 8 9]
    print("edges[:, 0]:", edges[:, 0])
    print("edges[:, 0].shape:", edges[:, 0].shape)
    print("edges[:, 1]:", edges[:, 1])
    print("edges[:, 1].shape:", edges[:, 0].shape)
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    print("adj行数：", adj.shape[0])
    print("adj列数：", adj.shape[1])
    # ww = open("adj.txt", "w", encoding="utf-8")
    # ww.write(str(adj))
    # adj行数： 10
    # adj列数： 10
    #
    # build symmetric adjacency matrix
    # 将非对称邻接矩阵转变为对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 打印消息：数据集有多少个节点、多少条边、每个样本有多少维特征
    print('Dataset has {} nodes, {} edges, {} dim of features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    # Dataset has 2708 nodes, 5429 edges, 1433 dims of features.
    # 返回特征的密集矩阵表示、邻接矩阵和标签的one-hot编码
    return features, adj, labels


# load_data()

# print("sys.maxsize:", sys.maxsize)   # 9223372036854775807


# with open("data/weibo/weibo.content", "r", encoding="utf-8")as weibo_content_file:
#     weibo_content = weibo_content_file.readlines()
#     print("len(weibo_content):", len(weibo_content))  # 72261
#
# with open("data/weibo/weibo.cites", "r", encoding="utf-8")as weibo_cites_file:
#     weibo_cites = weibo_cites_file.readlines()
#     print("len(weibo_cites):", len(weibo_cites))  # 68302


# 2021.4.3————重写load_data()
def load_data0(path="record/", dataset=""):   # ————————————————2021.4.2
    """Load citation network dataset (cora only for now)"""
    # str.format()函数用于格式化字符串
    print('Loading event {} ...'.format(dataset))
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype, delimiter, usecols, skip_header)
    # frame：文件名
    # dtype：数据类型
    # delimiter：分隔符
    # usecols：选择读哪几列，通常将属性集读为一个数组，将标签读为一个数组
    # skip_header：是否跳过表头
    idx_features_labels = np.genfromtxt("{}{}_check.txt".format(path, dataset), dtype=np.dtype(str))
    # print("idx_features_labels:", idx_features_labels)

    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    print("feature行数：", features.shape[0])   # ----->10
    print("feature列数：", features.shape[1])   # ----->256
    #
    # 提取样本的标签，并将其转换为one-hot编码形式
    labels = encode_onehot(idx_features_labels[:, -1])
    print("label行数：", labels.shape[0])
    print("label列数：", labels.shape[1])

    # 样本的id数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int64)
    print("idx:", idx)
    # # 由样本id到样本索引的映射字典
    # idx_map = {j: i for i, j in enumerate(idx)}  # i为索引，j为样本id。所以该字典为：{样本id:样本索引}
    # print("idx_map:", idx_map)

    return features.todense(), labels, idx


# feat1, lab1, idx1 = load_data0(dataset="3911188850873165")
# feat2, lab2, idx2 = load_data0(dataset="3680560520118168")
# idx_map1.update(idx_map2)
# print("idx:", idx_map1)
# print("idx1:", idx1)
# print("idx.type:", type(idx1))


# X, A, y = load_data(dataset="weibo")  # 特征、邻接矩阵、标签
# # 训练集样本标签、验证集样本标签、测试集样本标签、训练集索引列表
# # 验证集索引列表、测试集索引列表、训练数据的样本掩码
# y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
# print("y_train:", y_train)
# print("y_trian.type:", type(y_train))
# print("y_train.shape:", y_train.shape)


# 2021.4.5——生成只有源推文的数据
# 一、生成.content文件
def make_source_post_content():
    # 1.读取eid_list.txt
    eid_list_file = open("eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_list_file.readlines()
    for i in range(len(eid_list)):
        eid_list[i] = eid_list[i].replace("\n", "")

    # 2.读取record文件夹中的每一个txt文件，每个文件中的第一行即为源推文信息，然后存入文件中
    source_post_list = []
    source_post_file = open("source_post.content", "w", encoding="utf-8")
    for eid in eid_list:
        event_file = open("record/" + eid + "_check.txt", "r", encoding="utf-8")
        source_post = event_file.readlines()[0]
        source_post_list.append(source_post)

    for j in range(len(source_post_list)):
        if j != len(source_post_list) - 1:
            if source_post_list[j].split("\t")[0] == '3911188850873165' or source_post_list[j].split("\t")[0] \
                    == '3586313934109778':
                source_post_file.write(source_post_list[j] + "\n")
            else:
                source_post_file.write(source_post_list[j])
        else:
            source_post_file.write(source_post_list[j].replace("\n", ""))

    print("Make source post content file done!")


# 二、生成.cites文件
def make_source_post_cites():
    source_post_file = open("source_post.cites", "w", encoding="utf-8")

    # 1.读取eid_list.txt
    eid_list_file = open("eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_list_file.readlines()
    for ii in range(len(eid_list)):
        eid_list[ii] = eid_list[ii].replace("\n", "")

    # 2.生成.cites文件
    for eid in eid_list:
        if eid_list.index(eid) != len(eid_list) - 1:
            source_post_file.write(eid + "\t" + eid + "\n")
        else:
            source_post_file.write(eid + "\t" + eid)

    print("Make source post cites file done!")


# make_source_post_content()
# make_source_post_cites()


# ff = open("source_post.content", "r", encoding="utf-8")
# all_info = ff.readlines()
# bug_dict = {}
# for i in range(len(all_info)):
#     feat = all_info[i].split("\t")[1:-1]
#     if len(feat) > 256:
#         bug_dict[i] = all_info[i].split("\t")[0]
# print("bug_dict:", bug_dict)
# {1183: '3911188850873165', 3677: '3586313934109778'}
