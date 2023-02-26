# coding=UTF-8
# 模型执行器，进行模型创建，保存模型训练结果，加载模型和预测的功能
# 

'''
@File: execute.py
@Author: Wei Wei
@Time: 2022/7/30 16:30
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import os
import sys
import time
import tensorflow as tf
import seq2seqModel
from config import getConfig
import io
from progressbar import ProgressBar, Bar, Percentage
import numpy as np
import jieba
import pickle

config = getConfig.get_config()

vocab_input_size = config['encoder_vocab_size']
vocab_target_size = config['decoder_vocab_size']
embedding_dim = config['embedding_dim']
units = config['layer_size']
BATCH_SIZE = config['batch_size']
max_length_input, max_length_target = 20, 20


# 语句处理函数，在所有句子的开头和结尾分别加上start和end标识
def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    # print(w)
    return w


# 训练数据集处理函数，作用是读取文件中的数据，并进行初步的语句处理，在语句前后加上开始和结束标识。
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


# 计算最大语句长度，为避免特别长的语句出现，影响整体的计算复杂度和影响，取90%最长语句的长度作为最大语句长度。舍弃特别长的个别语句
def max_length(len_list):
    total_len = len(len_list)
    while True:
        max_len = np.max(len_list)
        avg_len = np.mean(len_list)
        if avg_len / max_len < 0.3 and len(len_list) / total_len > 0.9:
            del len_list[np.argmax(len_list)]
        else:
            break
    return np.max(len_list)


# 数据加载函数，可以根据需要按量加载数据
def read_data(path, num_examples):
    try:
        dataset_pkl = open(config['dataset'] + '/dataset.pkl', 'rb')
        input_tensor = pickle.load(dataset_pkl)
        input_tokenizer = pickle.load(dataset_pkl)
        target_tensor = pickle.load(dataset_pkl)
        target_tokenizer = pickle.load(dataset_pkl)
    except FileNotFoundError:
        print('start read sequence training data')
        input_lang, target_lang = create_dataset(path, num_examples)
        input_len = [len(sen.split(' ')) for sen in input_lang]
        target_len = [len(sen.split(' ')) for sen in target_lang]
        max_length_input = max_length(input_len)
        max_length_target = max_length(target_len)
        input_tensor, input_tokenizer = tokenize(input_lang, max_length_input)
        target_tensor, target_tokenizer = tokenize(target_lang, max_length_target)
        print('end read sequence training data')
        # 保存训练数据集
        dataset_pkl = open(config['dataset'] + '/dataset.pkl', 'wb')
        pickle.dump(input_tensor, dataset_pkl)
        pickle.dump(input_tokenizer, dataset_pkl)
        pickle.dump(target_tensor, dataset_pkl)
        pickle.dump(target_tokenizer, dataset_pkl)
        dataset_pkl.close()
    return input_tensor, input_tokenizer, target_tensor, target_tokenizer


# word2vec函数，通过统计训练集中的字符出现的频率，构建字典，并使用字典中的码值对训练集的语句进行替换
def tokenize(lang, max_length):
    # 实例化一个转换器，构建字典并使用字典中的码值对训练姐的语句进行替换
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config['encoder_vocab_size'], oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # 将训练语句的长度统一补全
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length, padding='post')

    return tensor, lang_tokenizer


# 加载数据集
input_tensor, input_token, target_tensor, target_token = read_data(config['seq_data'], config['max_train_data_size'])


def train():
    print("Preparing data in %s" % config['seq_data'])
    steps_per_epoch = len(input_tensor) // config['batch_size']
    print('训练总步数：{0}'.format(steps_per_epoch))
    encoder_hidden = seq2seqModel.encoder.initialize_hidden_state()
    checkpoint_dir = config['model_folder']
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print("reload pretrained model")
        seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    checkpoint_dir = config['model_folder']
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # start_time = time.time()

    print('Start epoch training!')
    progress = ProgressBar(maxval=steps_per_epoch, widgets=[Bar('=', '[', ']'), ' ', Percentage()])
    progress.start()
    epoch = 0
    while True:
        print('第{0}次训练：'.format(epoch))
        start_time_epoch = time.time()
        total_loss = 0
        percent = 0
        for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
            progress.update(int(batch * 100 / steps_per_epoch))
            batch_loss = seq2seqModel.train_step(input, target, target_token, encoder_hidden)
            total_loss += batch_loss
            # print(batch_loss.numpy())
            if percent < int(batch * 100 / steps_per_epoch):
                percent = int(batch * 100 / steps_per_epoch)
                print('progress: {0}%'.format(percent))
        # step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        # current_steps = +steps_per_epoch
        step_time_total = time.time() - start_time_epoch
        print('总耗时: {0}  平均每步loss {1:.4f}'.format(round(step_time_total), step_loss.numpy()))

        seq2seqModel.checkpoint.save(file_prefix=checkpoint_prefix)

        sys.stdout.flush()
        epoch += 1


def predict(sentence):
    checkpoint_dir = config['model_folder']
    seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    sentence = preprocess_sentence(sentence)
    # 对输入语句进行word2vec转换
    inputs = [input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    # 对输入语句按照最大长度补齐
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    encoder_output, encoder_hidden = seq2seqModel.encoder(inputs, hidden)
    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([target_token.word_index['start']], 0)
    for t in range(max_length_target):
        predictions, decoder_hidden, attention_weights = seq2seqModel.decoder(decoder_input, decoder_hidden,
                                                                              encoder_output)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if target_token.index_word[predicted_id] == 'end':
            break
        result += target_token.index_word[predicted_id] + ' '
        decoder_input = tf.expand_dims([predicted_id], 0)
    return result


if __name__ == '__main__':
    if len(sys.argv) - 1:
        config = getConfig.get_config(sys.argv[1])
    else:
        config = getConfig.get_config()
    print('\n>> Mode : %s\n' % (config['mode']))
    if config['mode'] == 'train':
        train()
    elif config['mode'] == 'serve':
        cont = True
        while cont:
            question = str(input("请问有什么咨询： "))

            if question == "exit":
                cont = False
            else:
                # 全模式分词，把句子中所有可以成词的词语都扫描出来，词语会重复，且不能解决歧义，适合关键词提取
                que_list = " ".join(jieba.cut(question))
                # 调用decode_line对生成回答信息
                answer = predict(que_list)
                answer = answer.replace(' ', '')
                print(answer)

