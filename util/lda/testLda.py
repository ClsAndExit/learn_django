# !/usr/bin/python3
# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from util.lda.testJieba import readfile
#从文件导入停用词表
stpwrdpath = "../sourceFile/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

res1 = readfile("../output_file/财经/10.txt")
print(res1)
res2 = readfile("../output_file/教育/11.txt")
print(res2)
res3 = readfile("../output_file/教育/10.txt")
print(res3)

corpus = [res1,res2,res3]
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)
print(cntTf)

lda = LatentDirichletAllocation(n_topics=2,learning_offset=50.,random_state=0)
docres = lda.fit_transform(cntTf)

print(docres)
print(lda.components_)