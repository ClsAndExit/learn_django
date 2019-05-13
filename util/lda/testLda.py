# !/usr/bin/python3
# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from util.lda.testJieba import readfile
from util.lda.testJieba import savefile

import os
#从文件导入停用词表
stpwrdpath = "../sourceFile/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'r')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

#用来记录语料库中所有类别文本的数量，该记录用于下文对文档-主题分布进行标签标注
file_count = list() #[90, 90, 90, 90, 89]
#为file_count的索引，标记当前记录的是哪个文件夹
index = 0
corpus = list()
corpus_path = "H:/PythonCode/learn_django/util/output_file/"  # 分词后分类预料库路径
catelist = os.listdir(corpus_path)  # 获取分词目录下所有子目录
for mydir in catelist:
    file_count.append(0)
    class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径
    file_list = os.listdir(class_path)  # 列举当前目录所有文件
    for file_path in file_list:
        file_count[index] += 1
        fullname = class_path + file_path  # 路径+文件名
        print("当前处理的文件是： ", fullname)
        content = readfile(fullname)  # 读取文件内容
        corpus.append(content)
    index += 1

#将文档数据集转化为单词矩阵
#如果不需要对特征选择进行分析，那么特征的数量就是被分析文档的数量的大小
cntVector = CountVectorizer(stop_words=stpwrdlst)
print(cntVector)
cntTf = cntVector.fit_transform(corpus)

lda = LatentDirichletAllocation(n_topics=5,learning_offset=50.,random_state=0)
docres = lda.fit_transform(cntTf)

#已知文本类别
file_type = 0
#文档的主题分布
for i in docres:
    arri = i.tolist()
    file_count[file_type] -= 1
    #进行先验标注
    arri.append(file_type)
    if file_count[file_type] == 0:
        file_type += 1

    savefile("../sourceFile/docres.txt",str(arri)+'\r',type=2)
# #主题、词分布
# for j in lda.components_:
#     arrj = j.tolist()
#     savefile("../sourceFile/components.txt",str(arrj)+'\r',type=2)