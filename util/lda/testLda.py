# !/usr/bin/python3
# -*- coding:utf-8 -*-
# import jieba
# jieba.suggest_freq('沙瑞金', True)
# jieba.suggest_freq('易学习', True)
# jieba.suggest_freq('王大路', True)
# jieba.suggest_freq('京州', True)
# with open('../sourceFile/nlp_test0.txt',encoding='utf-8') as f:
#     document = f.read()
#     document_decode = document
#     document_cut = jieba.cut(document_decode)
#     result = ' '.join(document_cut)
#     result = result.encode('utf-8')
#     with open("../sourceFile/nlp_test1.txt",'wb') as f2:
#         f2.write(result)
# f.close()
# f2.close()
#
# with open('../sourceFile/nlp_test2.txt',encoding='utf-8') as f:
#     document = f.read()
#     document_decode = document
#     document_cut = jieba.cut(document_decode)
#     result = ' '.join(document_cut)
#     result = result.encode('utf-8')
#     with open("../sourceFile/nlp_test3.txt",'wb') as f2:
#         f2.write(result)
# f.close()
# f2.close()
#
# with open('../sourceFile/nlp_test4.txt',encoding='utf-8') as f:
#     document = f.read()
#     document_decode = document
#     document_cut = jieba.cut(document_decode)
#     result = ' '.join(document_cut)
#     result = result.encode('utf-8')
#     with open("../sourceFile/nlp_test5.txt",'wb') as f2:
#         f2.write(result)
# f.close()
# f2.close()

#从文件导入停用词表
stpwrdpath = "../sourceFile/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

with open('../sourceFile/nlp_test1.txt',encoding='utf-8') as f3:
    res1 = f3.read()
print(res1)
with open('../sourceFile/nlp_test3.txt',encoding='utf-8') as f4:
    res2 = f4.read()
print(res2)
with open('../sourceFile/nlp_test5.txt',encoding='utf-8') as f5:
    res3 = f5.read()
print(res3)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
corpus = [res1,res2,res3]
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)
print(cntTf)

lda = LatentDirichletAllocation(n_topics=2,learning_offset=50.,random_state=0)
docres = lda.fit_transform(cntTf)

print(docres)
print(lda.components_)