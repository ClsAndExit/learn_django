#!/usr/bin/python3
# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from util.lda.testJieba import readfile
from util.lda.testJieba import savefile

import os
#打印每个主题下权重较高的term
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    #打印主题-词语分布矩阵
    print(model.components_)
def saveFile(doc,file_type):
    # 文档的主题分布
    for i in docres:
        arri = i.tolist()
        # 进行先验标注
        arri.append(file_type)
        savefile("../sourceFile/docres.txt", str(arri) + '\r', type=2)

#从文件导入停用词表
stpwrdpath = "../sourceFile/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'r')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

corpus_path = "H:/PythonCode/learn_django/util/output_file/"  # 分词后分类预料库路径
catelist = os.listdir(corpus_path)  # 获取分词目录下所有子目录
index = 0
for mydir in catelist:
    print("当前处理的类别是{0},该类别对应的索引值为{1}".format(mydir,index))
    index += 1
    corpus = list()
    class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径
    file_list = os.listdir(class_path)  # 列举当前目录所有文件
    for file_path in file_list:
        fullname = class_path + file_path  # 路径+文件名
        content = readfile(fullname)  # 读取文件内容
        corpus.append(content)
    cntVector = CountVectorizer(stop_words=stpwrdlst)
    cntTf = cntVector.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_topics=5, learning_offset=50., random_state=0)
    docres = lda.fit_transform(cntTf)
    print(docres)
    n_top_words = 20
    tf_feature_names = cntVector.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

    saveFile(docres,index)

    print("**************************************************************")