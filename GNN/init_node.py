import numpy as np
import json
# import jieba
# import jieba.analyse


# 1.将每个谣言的源+repost逐个提取放在一起，存放进文件名为“eid.txt”的文件中————2020.11.28
# 1.1提取————2020.12.6
# 1.1.1 先把eid_list提出来，其他的后边处理好之后再提取
def extract_eid():
    eid_list = []  # 事件id(根id)列表
    for line1 in open("Weibo.txt", 'r'):
        eid = line1.split()[0].split(":")[1]
        print("eid:", eid)
        eid_list.append(eid)
    print("eid_list:", eid_list)
    # 将list写入txt文件
    ff1 = open("data_structure/eid_list.txt", "w", encoding='utf-8')
    for i in range(len(eid_list)):

        if i != (len(eid_list)-1):
            ff1.write(eid_list[i] + "\n")
        else:
            ff1.write(eid_list[i])
    print("已生成eid_list.txt文件")


# 调用函数
# extract_eid()


# 1.1.2 把eid_label_dict提出来
def extract_eidlabel_dict():
    eid_list = []  # 事件id(根id)列表
    eid_label_dict = {}  # 事件id和标签的映射字典
    # eid_postid_dict = {}  # 事件id和repostid的映射字典

    for line1 in open("Weibo.txt", 'r'):
        eid = line1.split()[0].split(":")[1]
        # print("eid:", eid)
        label = line1.split()[1].split(":")[1]
        eid_list.append(eid)
        # print("eid_list:", eid_list)
        eid_label_dict[eid] = label
        s1 = line1.split("\t")[2]  # 字符串s1是每个事件的所有repost_id
        # print("每个事件的所有repost_id:", s1)

        # for s in range(len(s1.split()[1:])):
        #     eid_postid_dict[eid] = s1.split()[1:]
    # print("eid_list:", eid_list)
    # print("eid_label_dict:", eid_label_dict)
    # print("eid_postid_dict:", eid_postid_dict)
    print("extract_eidlabel_dict()提取id列表及映射字典完成")
    # 写入文件：

    ff2 = open("data_structure/eid_label_dict.json", "w")
    # ff3 = open("data_structure/eid_postid_dict.json", "w")
    # 将dict写入json
    eid_label_dict_js = json.dumps(eid_label_dict)
    ff2.write(eid_label_dict_js)
    # eid_postid_dict_js = json.dumps(eid_postid_dict)
    # ff3.write(eid_postid_dict_js)
    print("已生成eid_label_dict.json文件")
    # return eid_list, eid_label_dict, eid_postid_dict


# 1.2将同一个事件中的源推及其所有repost文本放在一个.txt文件中————2020.12.6
def gather_weiboset():
    # eid_list = extract_tweetid()[0]

    with open("data_structure/eid_list.txt", "r", encoding='utf-8')as t:
        eid_list = t.readlines()
        for n in range(len(eid_list)):
            eid_list[n] = eid_list[n].replace("\n", "")

        # print("eid_list:", eid_list)
    # eid_label_dict = extract_tweetid()[1]
    # eid_postid_dict = extract_tweetid()[2]
    for eid in eid_list:
        with open("Weibo/" + eid + ".json", 'r', encoding='utf-8')as f1:
            with open("weiboset/" + eid + ".txt", 'w', encoding='utf-8')as f2:
                list1 = json.load(f1)
                for i in range(len(list1)):
                    if i != (len(list1) - 1):
                        f2.write(list1[i]["text"] + "\n")
                    else:
                        f2.write(list1[i]["text"])
                    # text_postid_dict[list1[i]["text"]] = list1[i]["id"]

    # print("eid_list:", eid_list)
    # print("eid_label_dict:", eid_label_dict)
    # print("eid_postid_dict:", eid_postid_dict)
    print("gather_weiboset()集合文本完成")


# 调用函数：
# gather_weiboset()


# 2.将每个谣言txt文件中的空行和文本为“转发微博”的repost去除————2020.11.29
def remove_text():
    # eid_list = extract_tweetid()[0]
    eid_postid_dict = {}
    with open("data_structure/eid_list.txt", "r", encoding='utf-8')as t:   # 读取data_structure中的eid_list文件
        eid_list = t.readlines()
        for n in range(len(eid_list)):
            eid_list[n] = eid_list[n].replace("\n", "")
    # temp = []
    for i in range(len(eid_list)):

        with open("weiboset/" + eid_list[i] + ".txt", 'r', encoding='utf-8')as wt:  # 打开一个事件txt文件
            print("eid:", eid_list[i])
            temp = wt.readlines()   # temp为包括事件源文本和repost文本在内的所有文本列表
            for li in temp:
                # print("hhhhhhhh")
                if li == "\n" or li == "转发微博" or li == "转发微博。":

                    for line1 in open("Weibo.txt", 'r'):
                        if line1.split("\t")[0].split(":")[1] == eid_list[i]:
                            repoids = line1.split("\t")[2]   # 当前事件的所有repost_id
                            repoids_list = repoids.split(" ")[1:]
                            repoids_list.remove(repoids_list[temp.index(li)])
                            eid_postid_dict[eid_list[i]] = repoids_list
                    temp.remove(li)

                    # repoids_list = eid_postid_dict[eid_list[i]]  # 当前事件的所有repost_id

            with open("data_structure/eid_postid_dict.json", 'w')as ag:
                eid_postid_dict_js = json.dumps(eid_postid_dict)
                ag.write(eid_postid_dict_js)

                # eid_postid_dict_js = json.dumps(ag)
                # ag.write(eid_postid_dict_js)

            print("删除完成")
            with open("weiboset/" + eid_list[i] + ".txt", 'w', encoding='utf-8')as ww:
                for j in range(len(temp)):
                    ww.write(temp[j])
                    # if j != len(temp):
                    #     ww.write(temp[j] + "\n")
                    # else:
                    #     ww.write(temp[j])
                print("重生成文件操作完成")
    print("remove_text()去除空行和“转发微博”完成")


# 重新定义remove_text()函数

# 调用函数
# remove_text1()
# extract_eidlabel_dict()


# 重新定义remove_text()函数——version2——2020.12.22
def remove_text2(eid):
    # 读取eid相应的全部eid+repostid
    eid_postid_dict1 = {}
    repo_list = []
    with open("Weibo.txt", 'r', encoding='utf-8')as wb:
        for event in wb.readlines():
            if event.split("\t")[0].split(":")[1] == eid:
                eid_postid_dict1[eid] = event.split("\t")[2].split(" ")
                eid_postid_dict1[eid].remove("\n")
                print("eid_postid_dict:", eid_postid_dict1)
                # print("len(eid_postid_dict[eid]):", len(eid_postid_dict1[eid]))

    # 打开eid对应的txt文件并写入新文件
    with open("weiboset/" + eid + ".txt", 'r', encoding='utf-8')as wbe:
        with open("weiboset1/" + eid + ".txt", 'w', encoding='utf-8')as wbe1:
            all_texts = wbe.readlines()
            for ii in range(len(all_texts)):
                all_texts[ii] = all_texts[ii].replace("\n", "")
            print("all_texts:", all_texts)
            for text in range(len(all_texts)):
                print("当前repostid：", eid_postid_dict1[eid][text])
                print("当前repost：", all_texts[text])
                # print("len(all_texts):", len(all_texts))
                # if all_texts[text] != "转发微博" and all_texts[text] != "" and all_texts[text] != "转发微博。" and \
                #         all_texts[text] != "转发" and all_texts[text] != "轉發微博" and all_texts[text] != "轉發微博。":
                if "转发微博" not in all_texts[text] and "转发微博。" not in all_texts[text] and "转发" not in \
                        all_texts[text] and "转了" not in all_texts[text] and "轉發微博。" not in all_texts[text] \
                        and "轉發微博" not in all_texts[text] and " " not in all_texts[text]:
                    repo_list.append(eid_postid_dict1[eid][text])
                    if text != len(all_texts)-1:
                        wbe1.write(all_texts[text] + "\n")
                    else:
                        wbe1.write(all_texts[text])

                    # eid_postid_dict[eid].remove(eid_postid_dict[eid][all_texts.index(text) - 1])
    print("重写入当前事件文本文件完毕")
    eid_postid_dict1[eid] = repo_list
    eid_postid_dict1[eid] = eid_postid_dict1[eid][1:]
    print("新eid_postid_dict:", eid_postid_dict1)
    return eid_postid_dict1  # 返回当前事件的eid_postid_dict


# remove_text2('6633920051')
# 调用remove_text2()和extract_eidlabel_dict()来生成全部事件的新文件：————————————

def run_remove_text2_and_extract_eidlabel_dict():
    # 读取data_structure中的eid_list文件
    with open("data_structure/eid_list.txt", "r", encoding='utf-8')as t:
        eid_list = t.readlines()
        for n in range(len(eid_list)):
            eid_list[n] = eid_list[n].replace("\n", "")
        print("读取eid_list完成")
        print("eid_list:", eid_list)

    eid_postid_dict = {}
    for eid1 in eid_list:
        elem_eid_postid_dict = remove_text2(eid1)
        eid_postid_dict.update(elem_eid_postid_dict)

    # 将eid_postid_dict写入json文件:
    ff3 = open("data_structure/eid_postid_dict.json", "w")
    eid_postid_dict_js = json.dumps(eid_postid_dict)
    ff3.write(eid_postid_dict_js)
    print("已生成eid_postid_dict.json文件")

    # extract_eidlabel_dict()


# run_remove_text2_and_extract_eidlabel_dict()


# 调用函数：
# remove_text()
#
# print("eid_list:", eid_list)
# print("eid_label_dict:", eid_label_dict)
# print("eid_post_dict:", eid_postid_dict)


# 3.构造怀疑度特征————2020.12.14
# 3.1统计每个事件中的每个repost中所含怀疑词、非怀疑词的数量
def doubt_analyze(eid):
    # 将怀疑词列表列出
    doubt_words = ['真', '真的', '真的假的', '真假', '假的', '假', '是真', '是假', '是真是假', '真相', '假象', '真实', '真信', '真正',
                   '假冒', '假冒伪劣', '以假乱真', '骗', '骗人', '骗子', '骗术', '求证', '谣言', '证实', '求证实', '查证', '不相信',
                   '不信', '不信谣', '谣', '不传谣', '传谣']

    repo_count_dict = {}  # repost和怀疑词频、非怀疑词频以及单个repo总词数的映射字典
    eid_postid_dict_file = open("data_structure/eid_postid_dict.json", 'r', encoding='utf-8')
    eid_postid_dict2 = json.load(eid_postid_dict_file)  # 读取eid_postid_dict

    with open("weiboset/" + eid + ".txt", 'r', encoding='utf-8') as ff:   # 打开参数eid所表示的事件txt文件
        repo = ff.readlines()[1:]  # repo:当前事件的所有repost文本组成的列表,一个元素为一条repost
        for j in range(len(repo)):  # 对于repo中的每个元素（每条repost），每次循环：
            elem_repo = jieba.lcut(repo[j])  # 将每个repost进行分词
            print("elem_repo:", elem_repo)
            # repo[j] = elem_repo
            doubt_count = 0  # 怀疑词计数器
            no_doubt_count = 0  # 非怀疑词计数器
            # elem_repo_word_num = 0  # 单个repost中的词数
            for k in range(len(doubt_words)):
                for m in range(len(elem_repo)):
                    print("当前doubt_word:", doubt_words[k])
                    print("当前单词：", elem_repo[m])
                    if doubt_words[k] == elem_repo[m]:
                        doubt_count += 1
                    # else:
                    #     no_doubt_count += 1
                    # elem_repo_word_num += 1
            no_doubt_count = len(elem_repo) - doubt_count
            repo_count_dict[eid_postid_dict2[eid][j]] = {'doubt': doubt_count, 'no_doubt': no_doubt_count,
                                                         'repo_word_num': len(elem_repo)}
            # 将怀疑词数、非怀疑词数和当前repost所含词数放进字典
        print("repo_count_dict:", repo_count_dict)
        with open('repo_count_dict.json', 'w', encoding='utf-8')as w:
            r_js = json.dumps(repo_count_dict)
            w.write(r_js)
        return repo_count_dict, eid_postid_dict2


# 调用函数
# doubt_analyze('3488989674101466')


# 重写doubt_analyze():
def doubt_analyze1(eid):
    # 将怀疑词列表列出
    doubt_words = ['真', '真的', '真的假的', '真假', '假的', '假', '是真', '是假', '是真是假', '真相', '假象', '真实', '真信', '真正',
                   '假冒', '假冒伪劣', '以假乱真', '骗', '骗人', '骗子', '骗术', '求证', '谣言', '证实', '求证实', '查证', '不相信',
                   '不信', '不信谣', '谣', '不传谣', '传谣']

    repo_count_dict = {}  # repost和怀疑词频、非怀疑词频以及单个repo总词数的映射字典
    eid_postid_dict_file = open("data_structure/eid_postid_dict.json", 'r', encoding='utf-8')
    eid_postid_dict2 = json.load(eid_postid_dict_file)  # 读取eid_postid_dict

    with open("weiboset1/" + eid + ".txt", 'r', encoding='utf-8') as ff:   # 打开参数eid所表示的事件txt文件
        repo = ff.readlines()[1:]  # repo:当前事件的所有repost文本组成的列表,一个元素为一条repost
        for j in range(len(repo)):  # 对于repo中的每个元素（每条repost），每次循环：
            elem_repo = jieba.lcut(repo[j])  # 将每个repost进行分词
            print("当前elem_repo:", elem_repo)
            doubt_count = 0  # 怀疑词计数器
            for ele_word in doubt_words:  # 对于怀疑词列表中的每个怀疑词，每次循环：
                # print("当前doubt_word:", ele_word)
                doubt_count += repo[j].count(ele_word)  # 怀疑词计数器
            no_doubt_count = len(elem_repo) - doubt_count  # 非怀疑词计数器
            # 将怀疑词数、非怀疑词数和当前repost所含词数放进字典:
            repo_count_dict[eid_postid_dict2[eid][j]] = {'doubt': doubt_count, 'no_doubt': no_doubt_count,
                                                         'repo_word_num': len(elem_repo)}
        print("repo_count_dict:", repo_count_dict)
        with open("repo_count_dicts" + eid + ".json", 'w', encoding='utf-8')as w:
            r_js = json.dumps(repo_count_dict)
            w.write(r_js)
    print("完成怀疑度分析")
    return repo_count_dict, eid_postid_dict2


# doubt_analyze1('3556619570968102')


# 3.2使用doubt_analyze()中的repo_count_dict映射字典内容来求每个repost的怀疑度和一个事件的平均怀疑度
def doubt_feat(eid):
    # 调用函数
    repo_count_dict, eid_postid_dict = doubt_analyze1(eid)
    elem_repo_doubt_feat = {}  # 每个repost的怀疑度
    elem_repo_df_list = []  # 存放每个repost的怀疑度
    # event_doubt_feat = 0  # 当前事件的怀疑度特征
    for i in range(len(eid_postid_dict[eid])):
        # 求出每个repost的怀疑度
        elem_repo_doubt_feat[eid_postid_dict[eid][i]] = \
            (repo_count_dict[eid_postid_dict[eid][i]]['doubt'] - repo_count_dict[eid_postid_dict[eid][i]]['no_doubt']) / \
            repo_count_dict[eid_postid_dict[eid][i]]['repo_word_num']

        elem_repo_df_list.append(elem_repo_doubt_feat[eid_postid_dict[eid][i]])
    print("elem_repo_doubt_feat:", elem_repo_doubt_feat)
    sum1 = 0
    for j in range(len(elem_repo_df_list)):
        sum1 += elem_repo_df_list[j]
    if len(elem_repo_df_list) != 0:
        event_doubt_feat = sum1 / len(elem_repo_df_list)  # 当前事件的怀疑度特征
    else:
        event_doubt_feat = 0

    event_doubt_feat = event_doubt_feat.__round__(6)
    print("event_doubt_feat:", event_doubt_feat)
    return event_doubt_feat


# doubt_feat('3556619570968102')

# 调用doubt_feat()计算每个事件的怀疑度————2020.12.23
def run_doubt_feat():
    eid_df_dict = {}  # eid和doubt_feature的映射字典
    # 读取data_structure中的eid_list文件
    with open("data_structure/eid_list.txt", "r", encoding='utf-8')as t:
        eid_list1 = t.readlines()
        for n1 in range(len(eid_list1)):
            eid_list1[n1] = eid_list1[n1].replace("\n", "")
        print("读取eid_list完成")
        print("eid_list:", eid_list1)

    for ele_eid in eid_list1:
        eid_df_dict[ele_eid] = doubt_feat(ele_eid)
    print("eid_df_dict:", eid_df_dict)

    # 将eid_doubt_feat_dict写入json文件中
    with open("data_structure/eid_doubt_feat_dict.json", 'w')as edfd:
        eid_df_feat_js = json.dumps(eid_df_dict)
        edfd.write(eid_df_feat_js)


# run_doubt_feat()

# 怀疑度后续操作plan A———————————————————————————————2021.3.9
# 1.将eid与分词后的字词相映射。————2021.3.9
def eid_cleanwords_map():
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

# 2.判断bert词向量所表示的词是否在1的dict中。若在，则将eid对应的怀疑度添加到bert词向量中。
def add_doubtfeat_to_bertvec():
    bertvec_file = open("weibo_data_newbert_(-2).txt", "r", encoding="utf-8")
    eid_cleanwords_file = open("data_structure/eid_cleanword.json", "r", encoding="utf-8")
    eid_doubtfeat_file = open("data_structure/eid_doubt_feat_dict.json", "r", encoding="utf-8")
    bertword_file = open("data_structure/bertword.txt", "w", encoding="utf-8")

    eid_cleanwords = json.load(eid_cleanwords_file)
    eid_doubtft = json.load(eid_doubtfeat_file)

    bertvec_list = []
    for bertvec in bertvec_file.readlines():
        bertvec_list.append(bertvec.split(" ")[0])
    bertword_file.write(str(bertvec_list))
    # print("bertvec_list:", bertvec_list)


# add_doubtfeat_to_bertvec()
