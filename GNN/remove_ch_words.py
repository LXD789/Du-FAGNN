import sys
import nltk
from nltk.corpus import stopwords
import jieba
from utils import clean_str, clean_str_sst, loadWord2Vec

# ——————————————————这个文件要用到具体的数据集——————————————————————————————————————


# 下边临时加了注释   ——2020.9.20#####################################################################
# if len(sys.argv) < 2:     # 若sys.argv的长度<2,则退出解释器(退出状态为1，即失败)并打印提示文字
#     sys.exit("Use: python remove_words.py <dataset>")
# ######################################################################################################

# 下边临时加了注释   ——2020.9.20#####################################################################
# dataset = sys.argv[1]     # 定义dataset为sys.argv列表的第二个元素(即从程序外部输入的第一个元素)
# ####################################################################################################

# 下面是由于无法从终端手动指定数据集而做出的修改

dataset = 'weibo'

# 此处在使用中文数据集时需要暂时注释掉————2020.11.25
# if 'SST' in dataset:
#     func = clean_str_sst
#     # 若SST在dataset中(即从外部输入的数据集名称为SST)，则定义func为clean_str_sst函数的别名
# else:
#     func = clean_str
#     # 否则，定义func为clean_str函数的别名

# 下边临时加了注释   ——2020.9.20######################################################################
# noinspection PyBroadException
try:
    least_freq = sys.argv[2]
except Exception:
    least_freq = 1
    print('using default least word frequency = 5')
# ##########################################################################################################


# 此部分：显示停用词并去除停用词（这部分代码用于微博数据集）——2020.11.25
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
# print("stop_words:", stop_words)
stop_words_file = open("中文停用词库.txt", 'r')
stop_words = set(stop_words_file.readlines())
print("stop_words:", stop_words)


# 此部分：打开数据集文件，逐行读取，将每行的头尾字符去除并用latin1解码，
#         最后将去除字符并解码后的内容添加到doc_content_list(文件内容列表)中。
doc_content_list = []    # 文件内容列表，列表类型
with open('data/corpus/weibo/' + dataset + '.txt', 'rb') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode())
# print("doc_content_list:", doc_content_list)


# 此部分：计算每个词的词频
word_freq = {}  # to remove rare words   word_freq(词频)(字典类型)：为了去除出现次数很少的词

for doc_content in doc_content_list:
    words = jieba.lcut(doc_content)
    # print("使用jieba分词后：", words)
    # temp = func(doc_content)
    # words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

# 将分词后的词列表进行清洗(即将符合要求的词筛选出来)，并添加到clean_docs(文件清洗列表)
clean_docs = []   # 文件清洗列表
for doc_content in doc_content_list:
    words = jieba.lcut(doc_content)
    # temp = func(doc_content)
    # words = temp.split()
    doc_words = []
    for word in words:
        # 对于words列表中的每个元素：
        if dataset == 'mr' or 'SST' in dataset:
            doc_words.append(word)
        elif word not in stop_words and word_freq[word] >= least_freq:
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)
with open('data/corpus/' + dataset + '.clean.txt', 'w', encoding='utf-8') as f:
    f.write(clean_corpus_str)
# 打开文件，将clean_corpus_str写进文件中

# -------------------------------------------------------------------------------------------------------

# with open('data/corpus/' + dataset + '-new.clean.txt', 'w') as f:
#     f.write(clean_corpus_str)


path3 = "data/corpus/weibo-new.clean.txt"
path4 = "data/corpus/weibo.clean.txt"
# 下边为对.clean.txt文本进行去除空行操作

temp = []
space_label_id = []  # 存放空行对应的id
with open(path3, "r", encoding="utf-8") as c:
    with open(path4, "w", encoding="utf-8") as c1:
        lines = c.readlines()
        b = 0
        for line in lines:
            if line != "\n":
                c1.write(line)
            else:
                space_label_id.append(b)
                # print("行id：" + str(b))
            b += 1
        # print("完成")
# print("space_label_id:", space_label_id)
print("列表长度：", str(len(space_label_id)))


# 需要删除数据集文本文件中空行对应的label文件中的相应行
with open('data/weibo-old.txt', 'r', encoding='utf-8') as s:
    with open('data/weibo-temp.txt', 'w', encoding='utf-8') as n:
        s_lines = s.readlines()
        for s_line in s_lines:
            label_id = s_line.split("\t")[0]
            if len(space_label_id) != 0:
                if space_label_id[0] != int(label_id):
                    n.write(s_line)
                else:
                    space_label_id.pop(0)
            else:
                n.write(s_line)
        print("删空行id完成")

# 在删除空行的label之后，将label文件中的id刷新
with open('data/weibo-temp.txt', 'r', encoding='utf-8') as t:
    with open('data/weibo.txt', 'w', encoding='utf-8') as su:
        t_lines = t.readlines()
        j = 0
        for t_line in t_lines:
            # t_line.split("\t")[0] = str(j)
            # su.write(t_line)
            s1 = t_line.split("\t")[1]
            s2 = t_line.split("\t")[2]
            su.write(str(j) + "\t" + s1 + "\t" + s2)
            j += 1
        print("完成刷新id操作")
# ------------------------------------------------------------------------------------------------------


# 计算文件中每行文本包含的词语个数，并输出含元素最长行、含元素最短行和平均行
len_list = []   # 定义列表类对象len_list(长度列表)
with open('data/corpus/' + dataset + '.clean.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if line == '\n':
            continue
        temp = line.strip().split()
        len_list.append(len(temp))

print('min_len : ' + str(min(len_list)))
print('max_len : ' + str(max(len_list)))
print('average_len : ' + str(sum(len_list)/len(len_list)))
