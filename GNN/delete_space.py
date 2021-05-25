# path = "C:\\Users\\26298\\Desktop\\师姐程序\\twitter15.clean.txt"
# path2 = "C:\\Users\\26298\\Desktop\\师姐程序\\twitter15-new.txt"

# path = "C:\\Users\\Administrator\\Desktop\\twitter15-原\\txt文件\\twitter15.txt"   # 这是“twitter15-原”里的文本
path = "C:\\Users\\Administrator\\Desktop\\twitter15\\twitter15.txt"
path2 = "C:\\Users\\Administrator\\Desktop\\TextING-master\\data\\corpus\\twitter15.txt"
# 这是代码相应文件里的twitter15，接收path中去除空行后的数据

# 下边为对数据集文本进行去除空行操作
# with open(path2, "w", encoding='utf-8') as t:
#     with open(path, "r", encoding='utf-8') as f:
#         a = 0
#         lines = f.readlines()
#         for line in lines:
#             if line != "\n":
#                 t.write(line)
#             else:
#                 print("原数据集行id：" + a)
#                 a = a + 1

# ——————————————————————————————————————————————————————
# path3 = "C:\\Users\\Administrator\\Desktop\\TextING-master\\data\\corpus\\twitter15-new.clean.txt"
# path4 = "C:\\Users\\Administrator\\Desktop\\TextING-master\\data\\corpus\\twitter15.clean.txt"
# # 下边为对.clean.txt文本进行去除空行操作
#
# # temp = []
# space_label_id = []  # 存放空行对应的id
# with open(path3, "r", encoding="utf-8") as c:
#     with open(path4, "w", encoding="utf-8") as c1:
#         lines = c.readlines()
#         b = 0
#         for line in lines:
#             if line != "\n":
#                 c1.write(line)
#             else:
#                 space_label_id.append(b)
#                 print("行id：" + str(b))
#             b += 1
#         # print("完成")
# # print("space_label_id:", space_label_id)
# # print("列表长度：", str(len(space_label_id)))
#
#
# # 需要删除数据集文本文件中空行对应的label文件中的相应行
# with open('data/twitter15-old.txt', 'r', encoding='utf-8') as s:
#     with open('data/twitter15-temp.txt', 'w', encoding='utf-8') as n:
#         s_lines = s.readlines()
#         for s_line in s_lines:
#             label_id = s_line.split("\t")[0]
#             if space_label_id[0] != int(label_id):
#                 n.write(s_line)
#             else:
#                 space_label_id.pop(0)
#         print("删空行id完成")
#
# # 在删除空行的label之后，将label文件中的id刷新
# with open('data/twitter15-temp.txt', 'r', encoding='utf-8') as t:
#     with open('data/twitter15.txt', 'w', encoding='utf-8') as su:
#         t_lines = t.readlines()
#         j = 0
#         for t_line in t_lines:
#             # t_line.split("\t")[0] = str(j)
#             # su.write(t_line)
#             s1 = t_line.split("\t")[1]
#             s2 = t_line.split("\t")[2]
#             su.write(str(j) + "\t" + s1 + "\t" + s2)
#             j += 1
#         print("完成刷新id操作")
# —————————————————————————————————————————————————————————————

# with open(path, "r", encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         print(line)
# 以上为对twitter15-v1中的数据集所做的操作。————2020.10.26

# 下边是对twitter15-v2和twitter16-v2数据集所做的操作。————2020.10.26

# 1.去除source_tweets文件中的推文id
#    （1）twitter15-v2：
pa = "C:\\Users\\Administrator\\Desktop\\twitter15-v2\\source_tweets.txt"
pa2 = "C:\\Users\\Administrator\\Desktop\\twitter15-v2\\source_tweets-new.txt"

# with open(pa, 'r', encoding='utf-8') as p:
#     with open(pa2, 'w', encoding='utf-8') as p2:
#         pa_lines = p.readlines()
#         for pa_line in pa_lines:
#             pa_temp = pa_line.split('\t')[1]
#             p2.write(pa_temp)
#         print("完成去除twitter15-v2推文id操作")

#    （2）twitter16-v2：
pat = "C:\\Users\\Administrator\\Desktop\\twitter16-v2\\source_tweets.txt"
pat2 = "C:\\Users\\Administrator\\Desktop\\twitter16-v2\\source_tweets-new.txt"

# with open(pat, 'r', encoding='utf-8') as pp:
#     with open(pat2, 'w', encoding='utf-8') as pp2:
#         pa_lines = pp.readlines()
#         for pa_line in pa_lines:
#             pa_temp = pa_line.split('\t')[1]
#             pp2.write(pa_temp)
#         print("完成去除twitter16-v2推文id操作")

# 2.将label文件处理为程序所需要的格式
#   （1）twitter15-v2：
pa3 = "C:\\Users\\Administrator\\Desktop\\twitter15-v2\\label.txt"
pa4 = "C:\\Users\\Administrator\\Desktop\\twitter15-v2\\label-new.txt"

# with open(pa3, 'r', encoding='utf-8')as p3:
#     with open(pa4, 'w', encoding='utf-8')as p4:
#         p3_lines = p3.readlines()
#         for p3_line in p3_lines:
#             pure_label = p3_line.split(":")[0]
#             p4.write(pure_label + "\n")
#         print("完成处理twitter15-v2的label文件操作")

#   （2）twitter16-v2
pat3 = "C:\\Users\\Administrator\\Desktop\\twitter16-v2\\label.txt"
pat4 = "C:\\Users\\Administrator\\Desktop\\twitter16-v2\\label-new.txt"

# with open(pat3, 'r', encoding='utf-8')as pp3:
#     with open(pat4, 'w', encoding='utf-8')as pp4:
#         p3_lines = pp3.readlines()
#         for p3_line in p3_lines:
#             pure_label = p3_line.split(":")[0]
#             pp4.write(pure_label + "\n")
#         print("完成处理twitter16-v2的label文件操作")

# 下面代码是对.json文件的处理操作尝试—————2020.11.23——————————————————————————————————
# import jieba
import json

# with open("Weibo/4010312877.json", 'r', encoding='utf-8')as w:
#     json_data = json.load(w)
#     print("文件中数据：", json_data)
#     print("json_data类型：", type(json_data))

# with open("Weibo/1.json", 'a', encoding='utf-8')as jw:
#     # dict1 = {'a': 1, 'b': 2, 'c': 3}
#     dict1 = [1, 2, 3, 4]
#     json.dump(dict1, jw, ensure_ascii=False)

# ————————————————————————————————————————————————————————————


# 下面代码是对Weibo.txt和Weibo文件夹内数据的提取，以此来构造树字典————2020.11.24———————————————————
# from init_node import *
# import json
#
# treesDic = {}  # 全体树
# # treeDic = {}   # 一棵树
#
# nodes = []   # 存放节点的列表
# event_id = []  # 事件id(即根节点id)列表
# label = []     # 标签列表
# post_id = []   # repost的id列表
# # while
#
#
# with open("Weibo.txt", 'r', encoding='utf-8')as w1:
#     lines = w1.readlines()
#     for line in lines:
#         eid = line.split()[0].split(":")[1]
#         print("eid:", eid)
#         event_id.append(eid)
#         # lab = line.split()[1].split(":")[1]
#         # print("label:", lab)
#         node = Node()
#         node.eid = eid              # eid
#         # node.label = lab  # label
#         node.id = line.split()[2]   # id
#         node.parent = None          # parent
#
#         with open("Weibo/" + eid + ".json", 'r', encoding='utf-8')as w2:
#             list1 = json.load(w2)
#             # print("list1:", list1)
#             ind = 1
#             node.index = 0
#             node.text = list1[0]["text"]
#             # print("list1[1]:", list1[1])
#             cnode = Node()
#             treeDic = {}
#             for i in range(1, len(list1)):
#                 # print("i:", i)
#                 cnode.eid = eid
#                 cnode.id = list1[i]["id"]
#                 cnode.index = ind
#                 cnode.text = list1[i]["text"]
#                 cnode.parent = node
#                 node.children.append(cnode)
#                 nodes.append(cnode)
#                 ind += 1
#             treeDic["root"] = node
#             treeDic["post"] = nodes
#             # print("treeDic:", treeDic)
#             # treesDic[eid] = treeDic
#     print("treesDic:", treesDic)


# ————————————————————————————————————————————————————————————


# 下面是对Weibo.txt和Weibo文件夹下.json文件中的数据进行处理的代码————2020.11.25————————————————————
import json

# 1.将各谣言源提取出来：*********************************************************
# with open("Weibo.txt", 'r', encoding='utf-8')as w1:
#     lines = w1.readlines()
#     id_list = []   # 存放eid的列表
#     # label_list = []
#     id_label_dict = {}   # eid与label的映射字典
#     for line in lines:
#         eid = line.split()[0].split(":")[1]
#         print("eid:", eid)
#         label = line.split()[1].split(":")[1]
#         print("label:", label)
#         id_label_dict[eid] = label
#         id_list.append(eid)
#     with open("weibo_data.txt", 'w', encoding='utf-8')as wd:
#         with open("weibo_old_label.txt", 'w', encoding='utf-8')as wla:
#             count = 0
#             for i in range(len(id_list)):
#                 with open("Weibo/" + id_list[i] + ".json", 'r', encoding='utf-8')as wj:
#                     list1 = json.load(wj)
#                     if i != (len(id_list)-1):
#                         wd.write(list1[0]["text"] + "\n")
#                         if count <= 3732:   # 4664*80%,即80%的训练集，20%的测试集
#                             wla.write(str(i) + "\t" + "train" + "\t" + id_label_dict[id_list[i]] + "\n")
#                             count += 1
#                         else:
#                             wla.write(str(i) + "\t" + "test" + "\t" + id_label_dict[id_list[i]] + "\n")
#                             count += 1
#                     else:
#                         wd.write(list1[0]["text"])
#                         wla.write(str(i) + "\t" + "test" + "\t" + id_label_dict[id_list[i]])
#             print("已生成文件")

# 2.将1提取出来的数据进行分词
# import jieba
#
# with open("data/corpus/weibo_data.txt", 'r', encoding='utf-8')as wd:
#     all_text = wd.read()
#     # split_list = jieba.lcut(all_text, cut_all=True)
#     split_list = jieba.lcut(all_text)
#     print("分词后列表：", split_list)

# 处理“中文停用词库.txt”数据：**********************************************
import re

# with open("中文停用词库.txt" 操作")

# with open("中文停用词库-new.txt", 'r', encoding='utf-8')as w1:
#     lines = w1.readlines()
#     print("lines:", lines)

# with open("中文停用词库.txt", 'r')as w1:
#     lines = w1.readlines()
#     print("lines:", lines)

# 3.将使用bert生成的中文词向量进行提取：

# with open('wb-clean-bert-256-(-2).txt', 'r') as tcb:
#     with open('wb-bert256-(-2).txt', 'w', encoding='utf-8', errors='ignore')as tnb:
#         lines = tcb.readlines()
#         b = 1
#         a = 1
#         for line in lines:
#             # print("line:", line)
#             dict1 = eval(line)
#             # print("dict1:", dict1)
#             list1 = dict1["features"]
#             # print("list1:", list1)
#
#             for i in range(len(list1)):
#                 if list1[i]["token"] != "[CLS]" and list1[i]["token"] != "[SEP]":
#                     tnb.write(str(list1[i]["token"]) + " ")
#                     # print("词：", list1[i]["token"])
#                     list2 = list1[i]["layers"]
#                     # print("list2:", list2)
#                     temp_list_1 = list2[0]["values"]
#                     # print("temp_list_1:", temp_list_1)
#                     # temp_list_2 = list2[1]["values"]
#                     # print("temp_list_2", temp_list_2)
#                     # temp_list_3 = list2[2]["values"]
#                     # print("temp_list_3:", temp_list_3)
#                     # temp_list_4 = list2[3]["values"]
#                     # print("temp_list_4:", temp_list_4)
#                     # list_total = temp_list_4 + temp_list_3 + temp_list_2 + temp_list_1
#                     list_total = temp_list_1
#                     # list_total = [round(temp_list_1[i]+temp_list_2[i]+temp_list_3[i]+temp_list_4[i], 6)
#                     #     for i in range(min(len(temp_list_1), len(temp_list_2), len(temp_list_3), len(temp_list_4)))]
#                     # list_total = temp_list_1
#                     tnb.write(" ".join("%s" % elem for elem in list_total))
#                     tnb.write("\n")
#                     # print("list_total类型：", type(list_total))
#                     # print("list_total:", list_total)
#                     print("第" + str(a) + "个词向量")
#                     a = a + 1
#             print("第" + str(b) + "行文本")
#             b = b + 1
#         print("完成")

# ——————————————————————————————————————————————————————————————

# 下面是我的PLAN A1————2020.11.28————————————————————————————————————————————
import json
# from init_node import *
#
# eid_list = []   # 事件id(根id)列表
# eid_label_dict = {}   # 事件id和标签的映射字典
# eid_postid_dict = {}   # 事件id和repostid的映射字典
# # text_postid_dict = {}   # repostid和文本内容的映射字典
#
#
# # 1.将每个谣言的源+repost逐个提取放在一起，存放进文件名为“eid.txt”的文件中————2020.11.28
# def gather_weiboset():
#     # eid = None
#     # label = None
#
#     for line1 in open("Weibo.txt", 'r'):
#         eid = line1.split()[0].split(":")[1]
#         # print("eid:", eid)
#         label = line1.split()[1].split(":")[1]
#         eid_list.append(eid)
#         # print("eid_list:", eid_list)
#         eid_label_dict[eid] = label
#         s1 = line1.split("\t")[2]   # 字符串s1是每个事件的所有repost_id
#         s1_list = s1.split()
#         # print("line1.split(tab):", s1)
#
#         for s in range(len(s1_list)):
#             eid_postid_dict[eid] = s1_list
# #
#         with open("Weibo/" + eid + ".json", 'r', encoding='utf-8')as f1:
#             with open("weiboset/" + eid + ".txt", 'w', encoding='utf-8')as f2:
#                 list1 = json.load(f1)
#                 for i in range(len(list1)):
#                     if i != (len(list1)-1):
#                         f2.write(list1[i]["text"] + "\n")
#                     else:
#                         f2.write(list1[i]["text"])
#                     # text_postid_dict[list1[i]["text"]] = list1[i]["id"]
#     print("eid_list:", eid_list)
#     print("eid_label_dict:", eid_label_dict)
#     print("eid_postid_dict:", eid_postid_dict)
#     print("gather_weiboset()提取文本完成")
#     # return eid_list


# 2.将每个谣言txt文件中的空行和文本为“转发微博”的repost去除————2020.11.29
# def remove_text():
#     temp = []
#     for i in range(len(eid_list)):
#         # print("hhhhhhh")
#         with open("weiboset/" + eid_list[i] + ".txt", 'r', encoding='utf-8')as wt:
#             temp = wt.readlines()
#             for line in temp:
#                 if line == "\n" or "转发微博" in line:
#                     temp.remove(line)
#             # print("删除完成")
#             with open("weiboset1/" + eid_list[i] + ".txt", 'w', encoding='utf-8')as ww:
#                 for j in range(len(temp)):
#                     ww.write(temp[j])
#                     # if j != len(temp):
#                     #     ww.write(temp[j] + "\n")
#                     # else:
#                     #     ww.write(temp[j])
#                 # print("重生成文件操作完成")
#     print("remove_text()去除空行和“转发微博”完成")


# 2.1重写remove_text()————2021.3.21
def remove_text1():
    eid_list_file = open("data_structure/eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_list_file.readlines()
    for eid in eid_list:
        eid = eid.replace("\n", "")
        newall_lines = []
        with open("weiboset/" + eid + ".txt", 'r', encoding='utf-8')as f:
            all_lines = f.readlines()

            for line in all_lines:
                if line != "\n" and "转发微博" not in line and "转发微博\n" not in line and "轉發微博。\n" not in line and "转发微博。\n" not in line\
                        and "轉發微博\n" not in line and "转发\n" not in line and "转了" not in line and "转发" not in line:
                    newall_lines.append(line)
                    # all_lines.remove(line)
            with open("weiboset1/" + eid + ".txt", 'w', encoding='utf-8')as f1:
                for j in range(len(newall_lines)):
                    f1.write(newall_lines[j])
    print("remove_text()去除空行和“转发微博”完成")


# remove_text1()

# gather_weiboset()
# remove_text()
# #
# # print("eid_list:", eid_list)
# # print("eid_label_dict:", eid_label_dict)
# # print("eid_post_dict:", eid_postid_dict)
#
#
# # 3.构造推文图
# def construct_tweetgraph():
#     graph = {}
#     # 这一版本是将所有.txt文件构造的所有图（树）放进一个字典里
#     for i in range(len(eid_list)):
#         with open("weiboset/" + eid_list[i] + ".txt", 'r', encoding='utf-8')as f2:
#             lines = f2.readlines()
#             k = 0
#             for j in range(len(lines)):
#                 node = Node()
#                 node.eid = eid_list[i]
#                 # for k in range(len(eid_postid_dict[eid_list[i]])):
#                 #     node.id =
#
#                 # node.id = text_postid_dict[lines[j]]
#                 node.index = j
#                 node.text = lines[j]
#                 if j == 0:
#                     node.parent = None
#                     graph[j] = node
#                 else:
#                     node.parent = graph[0]
#                     node.id = eid_postid_dict[eid_list[i]][k]
#                     graph[0].children.append(node)
#                     graph[j] = node
#                     k += 1
#
#     print("construct_tweetgraph()构造图完成")
#     print("graph_dict:", graph)


# construct_tweetgraph()


# ——————————————————————————————————————————————————————————————

# 对情感词进行操作————2020.12.3—————————————————————————————————————————————
# f1 = open("C:\\Users\\Administrator\\Desktop\\正面情感词语（中文）.txt", 'r', encoding='utf-8-sig')
# f2 = open("C:\\Users\\Administrator\\Desktop\\正面情感词语（中文）1.txt", 'w', encoding='utf-8')
# lines = f1.readlines()
# ls = []
# for line in lines:
#     line = line.split()[0]
#     ls.append(line)
# print("ls:", ls)
# for i in range(len(ls)):
#     if i != len(ls)-1:
#         f2.write(ls[i] + " ")
#     else:
#         f2.write(ls[i])
#
# print("done!")

# ——————————————————————————————————————————————————————————————

# 统计源推文和repost的总个数————2020.12.6———————————————————————————————————————
# eid个数： 4664
# repost总数： 63415086
# f1 = open("Weibo.txt", 'r')
# lines = f1.readlines()
# eid_list = []  # 存放eid的列表
# eid_repostid_dict = {}  # eid和repostid映射字典
# eid_repocount_dict = {}  # eid和repocount映射字典
# repo_sum = 0   # repo计数器
# for i in range(len(lines)):
#     eid = lines[i].split()[0].split(":")[1]
#     eid_list.append(eid)
#     repo = (lines[i].split("\t")[2])[1:]
#     repo_count = len(repo)
#     eid_repocount_dict[eid] = repo_count
#     repo_sum += repo_count
#     eid_repostid_dict[eid] = repo
# print("eid个数：", len(eid_list))
# print("repost总数：", repo_sum)

# ——————————————————————————————————————————————————————————————

# 统计各事件的repost中怀疑词数和非怀疑词数——2020.12.12——————————————————————————————————
# doubt_words = ['真', '真的', '真的假的', '真假', '假的', '假', '是真', '是假', '是真是假', '真相', '假象', '真实', '真信', '真正',
#                '假冒', '假冒伪劣', '以假乱真', '骗', '骗人', '骗子', '骗术']
#
# repo_count_dict = {}  # repost和怀疑词频、非怀疑词频以及单个repo总词数的映射字典
# eid_postid_dict_file = open("data_structure/eid_postid_dict.json", 'r', encoding='utf-8')
# eid_postid_dict = json.load(eid_postid_dict_file)   # 读取eid_postid_dict
# eid_file = open("data_structure/eid_list.txt", 'r', encoding='utf-8')
# eids = eid_file.readlines()  # 读取所有eid
# for i in range(len(eids)):
#     with open("weiboset/" + eids[i] + ".txt", 'r', encoding='utf-8')as ff:   # 打开一个事件的txt文件
#         repo = ff.readlines()[1:]   # 当前事件的所有repost文本组成的列表
#         for j in range(len(repo)):
#             elem_repo = jieba.lcut(repo[j])   # 将每个repost进行分词
#             # repo[j] = elem_repo
#             doubt_count = 0  # 怀疑词计数器
#             no_doubt_count = 0  # 非怀疑词计数器
#             elem_repo_word_num = 0  # 单个repost中的词数
#             for k in range(len(doubt_words)):
#                 for m in range(len(elem_repo)):
#                     if doubt_words[k] in elem_repo[m]:
#                         doubt_count += 1
#                     else:
#                         no_doubt_count += 1
#                     elem_repo_word_num += 1
#             repo_count_dict[eid_postid_dict[eids[i]][j]] = {'doubt': doubt_count, 'no_doubt': no_doubt_count,
#                                                             'repo_word_num': elem_repo_word_num}
#             # 将怀疑词数、非怀疑词数和当前repost所含词数放进字典


# —————————————————————————————————————————————————————————————


# with open("Weibo.txt", 'r', encoding='utf-8')as f:
#     with open("Weibo.json", 'r', encoding='utf-8')as fs:
#         f2 = json.load()
#         print(f2)

# with open("WEI.txt", 'w', encoding='utf-8')as w:
#     f1 = f.readlines()
#     for i in range(len(f1)):
#         label1 = f1[i].split("\t")[1]
#         print(label1)
#         label = label1.split(":")[1]
#         w.write(label+"\n")
#
#
#
#
# print(label1)
# label = label1.split(":")[1]
# print(label)
# with open("WEI.txt", 'w', encoding='utf-8')as w:
#     w.write(label)


def eid_cleanwords_map():   # 生成eid_cleanword映射字典文件
    eid_file = open("data_structure/eid_list.txt", 'r', encoding='utf-8')
    cleanwords_file = open("data/corpus/weibo-new.clean.txt", 'r', encoding='utf-8')
    eid_cleanword_dict = {}
    eid_cleanword = open("data_structure/eid_cleanword.json", "w", encoding='utf-8')
    eids = eid_file.readlines()
    cleanwords = cleanwords_file.readlines()

    # 先看一下eid和cleanword中的行数是否相等:——>反馈：相等，都为4664
    print("len(eid_list):", len(eids))
    print("len(cleanword):", len(cleanwords))

    # 确定相等后，再将eid和cleanword进行映射
    for eve_eid in eids:
        # print("eve_eid:", eve_eid)
        # print("对应cleanword:", cleanwords[eids.index(eve_eid)])

        eid_cleanword_dict[eve_eid.rstrip("\n")] = cleanwords[eids.index(eve_eid)].rstrip("\n")

    eid_cw_dict_js = json.dumps(eid_cleanword_dict, ensure_ascii=False)
    eid_cleanword.write(eid_cw_dict_js)


# eid_cleanwords_map()


def add_doubtfeat_to_bertvec():
    # 1.提取bert向量中的字，集合为一个list文件——>bertword.txt
    bertvec_file = open("weibo_data_newbert_(-2).txt", "r", encoding="utf-8")  # 存储词和bert词向量
    bertword_file = open("data_structure/bertword.txt", "w", encoding="utf-8")   # 存储字词
    bertvec_list = []  # 存储字词的list
    for bertvec in bertvec_file.readlines():
        bertvec_list.append(bertvec.split(" ")[0])
    bertword_file.write(str(bertvec_list))
    # print("bertvec_list:", bertvec_list)

    # 2.构造word_doubtfeat映射字典文件
    weibotext_file = open("data/corpus/weibo-new.clean.txt", "r", encoding="utf-8")  # 微博源推文
    eid_doubtfeat_f = open("data_structure/eid_doubt_feat_dict.json", "r", encoding="utf-8")  # eid_怀疑度字典
    eid_cleanwd_f = open("data_structure/eid_cleanword.json", 'r', encoding="utf-8")  # eid_cleanword字典
    weibotext = weibotext_file.readlines()
    eid_df_dict = json.load(eid_doubtfeat_f)
    eid_cleanwd_dict = json.load(eid_cleanwd_f)

    # 将怀疑度与cleanwords以及单字词相关联
    word_doubtfeat_dict = {}  # 单字词-怀疑度映射字典

    doubtfeat_list = []  # 存储怀疑度的list
    for key in eid_df_dict:
        doubtfeat_list.append(eid_df_dict[key])
    # print("doubtfeatlist:", doubtfeat_list)

    cleanword_list = []   # cleanword列表
    for key in eid_cleanwd_dict:
        cleanword_list.append(eid_cleanwd_dict[key])
    # print("cleanword_list:", cleanword_list)

    # 构造word_doubtfeat映射字典
    for a in range(len(bertvec_list)):
        word_doubtfeat_dict[bertvec_list[a]] = 0.0
        for b in range(len(cleanword_list)):
            if cleanword_list[b].find(bertvec_list[a]) != -1:
                word_doubtfeat_dict[bertvec_list[a]] = doubtfeat_list[b]
                cleanword_list[b].replace(bertvec_list[a], "")
                break
            else:
                # word_doubtfeat_dict[bertvec_list[a]] = [0.0]
                continue
    print("done!")
    # print("word_doubtfeat_dict:", word_doubtfeat_dict)

    word_df_file = open("data_structure/word_doubtfeat_dict.json", "w")
    word_df_f_js = json.dumps(word_doubtfeat_dict, ensure_ascii=False)
    word_df_file.write(word_df_f_js)
    print("完成!")


# add_doubtfeat_to_bertvec()


# 将怀疑度合并到bert向量————2021.3.11
def together_bert_df():
    bertvec_file = open("weibo_data_newbert_(-2).txt", "r", encoding="utf-8")  # 存储词和bert词向量
    word_df_file = open("data_structure/word_doubtfeat_dict.json", "r", encoding="utf-8")  # word_doubtfeat_dict
    bertvec_list = bertvec_file.readlines()
    word_df_dict = json.load(word_df_file)
    new1bertvec_file = open("wbdata_newbert_(-2).txt", "w", encoding="utf-8")

    new1_bertvec = []

    for i in range(len(bertvec_list)):
        # print("bertvec_list:", bertvec_list[i])
        word = bertvec_list[i].split(" ")[0]
        # print("word:", word)
        new1bertvec_file.write(word + " ")

        vector = bertvec_list[i].split(" ")[1:]
        # vector[-1] = vector[-1].replace("\n", "")
        vector[len(vector) - 1] = vector[len(vector) - 1].replace("\n", "")
        # print("vector:", vector)
        vector.append(str(word_df_dict[word]))
        # print("vector1:", vector)
        new1bertvec_file.write(" ".join("%s" % elem for elem in vector))
        new1bertvec_file.write("\n")
        # new1_bertvec.append(vector.insert(0, word))

    # for j in range(len(vector)):
    #     new1bertvec_file.write(" ".join("%s" % elem for elem in vector))

    print("DONE!")


# together_bert_df()


def process():   # 2021.3.29
    eid_file = open("data_structure/eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_file.readlines()   # eid_list

    eid_postid_file = open("eid_postid_dict.json", "w", encoding="utf-8")
    eid_postid_dict = {}

    for eid in eid_list:
        repoids = []
        f1 = open("Weibo/" + eid.replace("\n", "") + ".json", "r", encoding="utf-8")
        f2 = open("weiboset1/" + eid.replace("\n", "") + ".txt", "r", encoding="utf-8")

        weibo_alltext = json.load(f1)
        new_weibo_all = f2.readlines()

        temp_dict = {}
        for i in range(1, len(weibo_alltext)):
            temp_dict[weibo_alltext[i]["id"]] = weibo_alltext[i]["text"]
        # print("temp_dict:", temp_dict)

        for j in range(len(new_weibo_all)):
            new_weibo_all[j] = new_weibo_all[j].replace("\n", "")

        no_dict = {}

        for k, v in temp_dict.items():
            if v in new_weibo_all:
                repoids.append(k)
            else:
                no_dict[k] = v
                continue

        eid_postid_dict[eid.replace("\n", "")] = repoids
        # print("no_dict:", no_dict)
        hh = open("no_dicts/" + eid.replace("\n", "") + "_nodict.json", "w", encoding="utf-8")
        hh_js = json.dumps(no_dict)
        hh.write(hh_js)

    eid_postid_js = json.dumps(eid_postid_dict)
    eid_postid_file.write(eid_postid_js)


# process()


# 计算每个事件中的repost个数,并找出最小的repo_num值————2021.4.1
def count_repo_num():
    eid_list_file = open("data_structure/eid_list.txt", "r", encoding="utf-8")
    eid_list = eid_list_file.readlines()

    repo_num_file = open("repo_num.json", "w", encoding="utf-8")
    repo_num_file1 = open("repo_num.txt", "w", encoding="utf-8")

    repo_num = []
    repo_num_dict = {}
    min_repo_num = 0
    for eid in eid_list:
        event_file = open("weiboset1/" + eid.replace("\n", "") + ".txt", "r", encoding="utf-8")
        event = event_file.readlines()
        repo_num.append(len(event) - 1)
        repo_num_dict[eid.replace("\n", "")] = len(event) - 1
        if (len(event) - 1) < min_repo_num:
            min_repo_num = len(event) - 1

    repo_js = json.dumps(repo_num_dict)
    repo_num_file.write(repo_js)
    repo_num_file1.write(str(repo_num))
    print("min_repo_num:", min_repo_num)


# count_repo_num()   # 3911188850873165
