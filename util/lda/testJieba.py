#@Time    :2018/12/28 19:50
import os
import jieba

# 保存文件的函数
def savefile(savepath,content,type=1):
    if type == 1:
        fp = open(savepath,'w',encoding='GBK',errors='ignore')
    else:
        fp = open(savepath,'a',encoding='GBK',errors='ignore')
    fp.write(content)
    fp.close()

# 读取文件的函数
def readfile(path):
    fp = open(path, "r", encoding='GBK', errors='ignore')
    content = fp.read()
    fp.close()
    return content

if __name__=="__main__":
    corpus_path = "E:\\DownloadFile\\BDWP\\sougou_all\\xinwen\\"  # 未分词分类预料库路径
    seg_path = "H:/PythonCode/learn_django/util/output_file/"  # 分词后分类语料库路径
    catelist = os.listdir(corpus_path)  # 获取未分词目录下所有子目录
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径
        seg_dir = seg_path + mydir + "/"  # 拼出分词后预料分类目录
        if not os.path.exists(seg_dir):  # 是否存在，不存在则创建
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path)  # 列举当前目录所有文件
        for file_path in file_list:
            fullname = class_path + file_path  # 路径+文件名
            print("当前处理的文件是： ", fullname)
            content = readfile(fullname).strip()  # 读取文件内容
            content = content.replace("\n", "").strip()  # 删除换行和多余的空格
            content_seg = jieba.cut(content)  # jieba分词
            result = ' '.join(content_seg)
            savefile(seg_dir + file_path, result)  # 保存



