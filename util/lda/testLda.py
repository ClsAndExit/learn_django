# !/usr/bin/python3
# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from util.lda.testJieba import readfile
from util.lda.testJieba import savefile

import os
#从文件导入停用词表
stpwrdpath = "../sourceFile/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

corpus = list()
corpus_path = "H:/PythonCode/learn_django/util/output_file/"  # 分词后分类预料库路径
catelist = os.listdir(corpus_path)  # 获取分词目录下所有子目录
for mydir in catelist:
    class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径
    file_list = os.listdir(class_path)  # 列举当前目录所有文件
    for file_path in file_list:
        fullname = class_path + file_path  # 路径+文件名
        print("当前处理的文件是： ", fullname)
        content = readfile(fullname)  # 读取文件内容
        corpus.append(content)
# res1 = readfile("../output_file/财经/10.txt")
# print(res1)
# res2 = readfile("../output_file/教育/11.txt")
# print(res2)
# res3 = readfile("../output_file/教育/10.txt")
# print(res3)
# corpus = [res1,res2,res3]

#将文档数据集转化为单词矩阵
#如果不需要对特征选择进行分析，那么特征的数量就是被分析文档的数量的大小
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)

lda = LatentDirichletAllocation(n_topics=5,learning_offset=50.,random_state=0)
docres = lda.fit_transform(cntTf)

#文档的主题分布
for i in docres:
    savefile("../sourceFile/docres.txt",str(i)+'\r',type=2)
#主题、词分布
for j in lda.components_:
    savefile("../sourceFile/components.txt",str(j)+'\r',type=2)