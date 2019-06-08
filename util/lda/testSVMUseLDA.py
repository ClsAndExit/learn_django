#!/usr/bin/python3
# -*- coding:utf-8 -*-
from sklearn import svm
import random
from sklearn import metrics
import numpy
#调整了格式，一行是一条数据
def inputdata(filename):
    f = open(filename,'r',encoding='utf-8')
    linelist = f.readlines()
    return linelist

def splitset(trainset,testset,split_num):
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    for i in trainset:
        data = eval(i)
        train_words.append(data[0:split_num])
        train_tags.append(data[split_num])

    for i in testset:
        data = eval(i)
        test_words.append(data[0:split_num])
        test_tags.append(data[split_num])

    return train_words,train_tags,test_words,test_tags

#按比例划分训练集与测试集
def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet = []
    copy = dataset
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return trainSet,copy

#得到准确率和召回率
def evaluate(actual, pred):
    m_accuracy=metrics.accuracy_score(actual,pred)
    m_precision = metrics.precision_score(actual, pred,average='macro')
    m_recall = metrics.recall_score(actual,pred,average='macro')
    print('accuracy:{0:0.3f}'.format(m_accuracy))
    print('precision:{0:0.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))

#创建svm分类器
def train_clf(train_data, train_tags):
    clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    clf.fit(train_data, numpy.asarray(train_tags))

    return clf

if __name__ == '__main__':
    linelist = inputdata('../sourceFile/docres.txt')
    # 划分成两个训练集和测试集
    trainset, testset = splitDataset(linelist, 0.80)
    print('train number:', len(trainset))
    print('test number:', len(testset))

    train_words, train_tags, test_words, test_tags = splitset(trainset, testset,split_num=12)
    clf = train_clf(train_words,train_tags)
    re = clf.predict(test_words)
    evaluate(numpy.asarray(test_tags),re)
