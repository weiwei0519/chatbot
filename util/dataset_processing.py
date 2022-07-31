# coding=UTF-8
# 对问答预料进行预处理
# 

'''
@File: dataset_processing.py
@Author: Wei Wei
@Time: 2022/7/30 11:10
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

from config import getConfig
import os
from progressbar import ProgressBar
import jieba

config = getConfig.get_config()
chat_ds_file = config['chat_dataset']
if not os.path.exists(chat_ds_file):
    exit()

print('start loading chat dataset')
chat_ds = []  # 用于存储对话的列表
with open(chat_ds_file, encoding='utf-8') as f:
    chat = []  # 存储一次完整对话
    for line in f:
        line = line.strip('\n').replace('/', '')  # 去除换行符，并将原文件中已经分词的标记去掉，重新用结巴分词.
        if line == '':
            continue
        if line[0] == config['e']:
            if chat:
                chat_ds.append(chat)
            chat = []
        elif line[0] == config['m']:
            chat.append(line.split(' ')[1])  # 将一次完整的对话存储下来
    f.close()
print('finish loading chat dataset')

print('start processing chat dataset')
progress = ProgressBar().start()
# 对训练集的对话进行分类，分为问和答，或者叫上文、下文，这个主要是作为encoder和decoder的熟练数据
seq = []
for index, chat in enumerate(chat_ds):
    if len(chat) == 1:
        continue
    if len(chat) % 2 != 0:  # 因为默认是一问一答的，所以需要进行数据的粗裁剪，对话行数要是偶数的
        chat = chat[:-1]
    for i in range(len(chat)):
        if i % 2 == 0:
            chat[i] = " ".join(jieba.cut(chat[i]))  # 使用jieba分词器进行分词
            chat[i + 1] = " ".join(jieba.cut(chat[i + 1]))
            # 问答句的数据结构为：问句分词 + ‘\t’ + 答句分词，偶数行为问句（i为偶数），奇数行为答句（i为奇数）
            seq.append(chat[i] + '\t' + chat[i + 1])
    progress.update(int(index * 100 / len(chat_ds)))
progress.finish()
print('finish processing chat dataset')

print('start archiving chat dataset')
progress = ProgressBar().start()
seq_train = open(config['seq_data'], 'w', encoding='UTF-8')
for i in range(len(seq)):
    seq_train.write(seq[i] + '\n')
    progress.update(int(i * 100 / len(seq)))
seq_train.close()
progress.finish()
print('finish archiving chat dataset')
