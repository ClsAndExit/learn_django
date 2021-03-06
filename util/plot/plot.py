# encoding=utf-8
from matplotlib import pyplot as plt
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import *

fonts = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)
fonts1 = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=12)


##C:\Windows\Fonts\simsun.ttc
def zhexian():
    start = 50  # 起始特征数量
    feature = 500  # 选取特征词个数
    length = 50  # 每次增长特征数量
    # 指定图形的字体 fontdict=myfont
    myfont = {'family': 'Times New Roman',
              'color': 'darkred',
              'weight': 'normal',
              'size': 20,
              }

    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    mpl.rc('font', family='Times New Roman', size=12)

    # plt.title("NB")
    plt.xlabel('主题数K', fontsize=12, color='black',fontproperties=fonts)
    plt.ylabel('F1值', fontsize=12, color='black',fontproperties=fonts)
    plt.title("不同主题数K对应的F1值",fontproperties=fonts)

    # 画网格
    # 将x主刻度标签设置为feature/10的倍数(也即以feature/10为主刻度单位其余可类推)
    xmajorLocator = MultipleLocator(feature / 10);
    # 设置x轴标签文本的格式
    xmajorFormatter = FormatStrFormatter('%i')  # '%3.1f'
    # 将x轴次刻度标签设置为feature/len的倍数
    xminorLocator = MultipleLocator(length)
    # 设置主刻度标签的位置,标签文本的格式
    ax = subplot(111)
    t = arange(start, feature + 1, length)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    # 显示次刻度标签的位置,没有标签文本
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度 #,linewidth = "0.2"
    ax.xaxis.grid(True, which='minor', linewidth="0.15")
    plt.grid(True, axis='y', color="Black", linewidth="0.15")  #

    scores1 = [
        0.667898306, 0.768574681, 0.859711879, 0.936929839, 0.870172856, 0.810305676, 0.750058312, 0.699605647,
        0.677482211, 0.630768444]
    scores2 = [
        0.789783066, 0.857846817, 0.871918795, 0.992698390, 0.917028567, 0.830056761, 0.805083129, 0.760956479,
        0.748722117, 0.676084447]

    # 折线图
    # plt.scatter(np.arange(0,len(scores)),scores)  #散点图 plt.bar()条形图
    plot(t, scores1, label='LDA', marker='o', color='#FFD700', linewidth="2")
    plot(t, scores2, label='Huffman-LDA', marker='*', color='#87CEFF', linewidth="2")

    # 调节轴之间的间距和轴与边框之间的距离
    plt.subplots_adjust(left=.13)
    plt.subplots_adjust(right=.97)
    plt.subplots_adjust(top=.97)
    plt.subplots_adjust(bottom=.12)

    # plt.legend(ncol=2,loc=8)  #ncol=2显示两列,显示线条label
    plt.legend(ncol=2, loc=3,prop=fonts1)  ##cmr10:宋体，cmb10：timesnewroman
    plt.show()


def zhuzhaungtu1_1():

    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    mpl.rc('font', family='Times New Roman', size=12)
    # X轴数据

    num_list = [0.891,0.895,0.903,0.879,0.915]

    num_list2 = [0.827,0.855,0.872,0.814,0.834]

    # X轴标签
    classifier = ('互联网', '商业', '教育', '娱乐', '体育')

    indices = np.arange(len(num_list))
    rects1 = plt.bar(indices - .15, num_list, .3, label="Huffman-LDA+SVM",
                     color='navy')  # 这里是产生横向柱状图 barh h--horizontal，bar纵向
    rects2 = plt.bar(indices + .15 , num_list2, .3, label="Huffman-LDA+NB", color='c')

    plt.ylim(ymax=1.0, ymin=0.5)
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.ylabel("精确率",fontproperties=fonts)  # X轴标签

    x_pos = np.arange(len(classifier))
    plt.xticks(x_pos, classifier, fontsize=12,fontproperties=fonts)
    plt.title("应用不同分类器得出各类别精确率",fontproperties=fonts)
    plt.legend(loc='best', fontsize=10,prop=fonts1)  # 参数loc=表示图例放置的位置，左右上角
    plt.subplots_adjust(left=.12)  # 调节轴之间的间距和轴与边框之间的距离
    plt.subplots_adjust(right=.97)
    plt.subplots_adjust(top=.97)
    plt.subplots_adjust(bottom=.06)

    plt.grid(True, axis='y', color="Black",
             linewidth="0.15")  # grid 表示是否显示图轴网格,b=None|True|False|on|off,是否开启, which=’major’，minor|both 选择主、次网格开启方式, axis=’both’，x|y 选择使用网格的数轴  #696969

    plt.show()

def zhuzhaungtu1_2():
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    mpl.rc('font', family='Times New Roman', size=12)
    # X轴数据
    num_list = [0.896,0.868,0.902]

    num_list2 = [0.843,0.878,0.822]


    # X轴标签
    classifier = ('Macro-P', 'Macro-R', 'Macro-F1')

    indices = np.arange(len(num_list))
    rects1 = plt.bar(indices - .15, num_list, .3, label="Huffman-LDA+SVM",
                     color='navy')  # 这里是产生横向柱状图 barh h--horizontal，bar纵向
    rects2 = plt.bar(indices + .15 , num_list2, .3, label="Huffman-LDA+NB", color='c')

    plt.ylim(ymax=1.0, ymin=0.5)
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.ylabel("宏平均值",fontproperties=fonts)  # X轴标签

    x_pos = np.arange(len(classifier))
    plt.xticks(x_pos, classifier, fontsize=12,fontproperties=fonts1)
    plt.title("应用不同分类器对应的宏平均值",fontproperties=fonts)
    plt.legend(loc='best', fontsize=10,prop=fonts1)  # 参数loc=表示图例放置的位置，左右上角
    plt.subplots_adjust(left=.12)  # 调节轴之间的间距和轴与边框之间的距离
    plt.subplots_adjust(right=.97)
    plt.subplots_adjust(top=.97)
    plt.subplots_adjust(bottom=.06)

    plt.grid(True, axis='y', color="Black",
             linewidth="0.15")  # grid 表示是否显示图轴网格,b=None|True|False|on|off,是否开启, which=’major’，minor|both 选择主、次网格开启方式, axis=’both’，x|y 选择使用网格的数轴  #696969

    plt.show()


def zhuzhaungtu2_1():
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    mpl.rc('font', family='Times New Roman', size=12)
    # X轴数据

    num_list = [0.748,0.636,0.563,0.673,0.736]
    num_list2 = [0.801,0.695,0.701,0.777,0.817]
    num_list3=[0.891,0.895,0.903,0.879,0.915]

    # X轴标签  LR 、KNN、NB、RF、SVM
    classifier = ('互联网', '商业', '教育', '娱乐', '体育')

    indices = np.arange(len(num_list))
    rects1 = plt.bar(indices - .3, num_list, .3, label="SVM",
                     color='navy')  # 这里是产生横向柱状图 barh h--horizontal，bar纵向
    rects2 = plt.bar(indices , num_list2, .3, label="LDA+SVM", color='c')
    rects3 = plt.bar(indices + .3, num_list3, .3, label="Huffman-LDA+SVM", color='#FFD700')

    plt.ylim(ymax=1.0, ymin=0.5)
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.ylabel("精确率",fontproperties=fonts)  # X轴标签

    x_pos = np.arange(len(classifier))
    plt.xticks(x_pos, classifier, fontsize=12,fontproperties=fonts)
    plt.title("不同类别精确率",fontproperties=fonts)
    plt.legend(loc='best', fontsize=10,prop=fonts1)  # 参数loc=表示图例放置的位置，左右上角
    plt.subplots_adjust(left=.12)  # 调节轴之间的间距和轴与边框之间的距离
    plt.subplots_adjust(right=.97)
    plt.subplots_adjust(top=.97)
    plt.subplots_adjust(bottom=.06)

    plt.grid(True, axis='y', color="Black",
             linewidth="0.15")  # grid 表示是否显示图轴网格,b=None|True|False|on|off,是否开启, which=’major’，minor|both 选择主、次网格开启方式, axis=’both’，x|y 选择使用网格的数轴  #696969

    plt.show()

def zhuzhaungtu2_2():
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    mpl.rc('font', family='Times New Roman', size=12)
    # X轴数据
    num_list = [0.671,0.636,0.691]
    num_list2 = [0.758,0.763,0.869]
    num_list3=[0.896,0.868,0.902]

    # X轴标签
    classifier = ('Macro-P', 'Macro-R', 'Macro-F1')

    indices = np.arange(len(num_list))
    rects1 = plt.bar(indices - .3, num_list, .3, label="SVM",
                     color='navy')  # 这里是产生横向柱状图 barh h--horizontal，bar纵向
    rects2 = plt.bar(indices , num_list2, .3, label="LDA+SVM", color='c')
    rects3 = plt.bar(indices + .3, num_list3, .3, label="Huffman-LDA+SVM", color='#FFD700')


    plt.ylim(ymax=1.0, ymin=0.5)
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.ylabel("宏平均值",fontproperties=fonts)  # X轴标签

    x_pos = np.arange(len(classifier))
    plt.xticks(x_pos, classifier, fontsize=12,fontproperties=fonts1)
    plt.title("使用LDA主题模型对应的宏平均值",fontproperties=fonts)
    plt.legend(loc='best', fontsize=10,prop=fonts1)  # 参数loc=表示图例放置的位置，左右上角
    plt.subplots_adjust(left=.12)  # 调节轴之间的间距和轴与边框之间的距离
    plt.subplots_adjust(right=.97)
    plt.subplots_adjust(top=.97)
    plt.subplots_adjust(bottom=.06)

    plt.grid(True, axis='y', color="Black",
             linewidth="0.15")  # grid 表示是否显示图轴网格,b=None|True|False|on|off,是否开启, which=’major’，minor|both 选择主、次网格开启方式, axis=’both’，x|y 选择使用网格的数轴  #696969

    plt.show()

if __name__ == '__main__':
# zhexian()
# zhuzhaungtu1_1() #图4-8
    zhuzhaungtu1_2() #图4-9
    #zhuzhaungtu2_1()  #图4-6
# zhuzhaungtu2_2() #图4-7