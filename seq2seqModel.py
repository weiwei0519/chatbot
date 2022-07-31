# coding=UTF-8
# 按照Encoder-Decoder框架构建一个完整的Seq2Seq模型
# 

'''
@File: seq2seqModel.py
@Author: Wei Wei
@Time: 2022/7/30 11:25
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import tensorflow as tf
from config import getConfig

config = getConfig.get_config()


# 定义encoder模型，Encoder是keras Model的一个子类
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units  # encoder神经元数量
        # 定义Embedding层，对输入序列进行向量化，防止特征稀疏
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # 模型采用GRU结构
        self.GRU = tf.keras.layers.GRU(self.encoder_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    # 定义采用GRU为RNN的网络结构，调用自身的方法
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.GRU(x, initial_state=hidden)
        return output, state

    # 定义初始化隐藏层状态，用于初始化隐藏层的神经元
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))


# 定义Attention机制模型
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # 初始化定义权重网络层W1、W2以及最后的打分网络层V，最终打分结果作为注意力的权重值
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # 输入、输出逻辑变换调用函数，调用自身的方法
    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 构建评价计算网络机构，首先计算w1和w2，然后将w1和w2的和进行tanh进行非线性变换，最后输入打分网络层中。
        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # 计算attention_weights的值，使用softmax将score的值进行归一化，得到的是总和唯一的各个score值的概率分布
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# 定义decoder模型，为keras.Model的子类
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units  # decoder神经元的数量
        # 定义Embedding层，对输入序列进行向量化，防止特征稀疏
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # 模型采用GRU结构
        self.GRU = tf.keras.layers.GRU(self.decoder_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # 初始化定义全连接输出层
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.decoder_units)

    # 输入、输出逻辑变换调用函数
    def call(self, x, hidden, enc_output):
        # 解码器输出的维度是(batch_size, max_length, hidden_size)
        # 根据输入hidden和输出值使用Attention机制计算文本向量和注意力权重，hidden就是编码器输出的编码向量
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # 对解码器的输入进行Embedding处理
        x = self.embedding(x)
        # 将Embedding之后的向量和经过Attention后的编码器输出的编码向量进行链接，然后作为输入向量输入到GRU模型中。
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.GRU(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


vocab_input_size = config['encoder_vocab_size']
vocab_target_size = config['decoder_vocab_size']
embedding_dim = config['embedding_dim']
units = config['layer_size']
BATCH_SIZE = config['batch_size']

encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

# 定义优化器
optimizer = tf.keras.optimizers.Adam()
# 定义损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss_function(real, pred):
    # 为了增强训练效果和提高泛化性，将训练数据中最常用的词遮罩，需构建一个mask向量
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    # 计算平均损失值
    return tf.reduce_mean(loss_)


checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


# 训练方法，对输入数据集进行一次循环训练
def train_step(input, target, target_lang, encoder_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        # 构建编码器
        encoder_output, encoder_hidden = encoder(input, encoder_hidden)
        decoder_hidden = encoder_hidden
        # 构建解码器输入向量，首词使用start对应的字典码值作为向量的第一个数值，维度是batch_size大小，也就是一次批量训练的语句数量
        decoder_input = tf.expand_dims([target_lang.word_index['start']] * BATCH_SIZE, 1)
        for t in range(1, target.shape[1]):
            # 将构建的编码器输入向量和编码器输出对话中上一句的编码向量作为输入，输入解码器中，训练解码器
            predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
            # 计算损失值
            loss += loss_function(target[:, t], predictions)
            # 将对话中的下一句逐步分时作为编码器的输入，相当于进行位移输入，先从start标识开始，逐步输入对话中的下一句
            decoder_input = tf.expand_dims(target[:, t], 1)
    # 计算批处理的平均损失
    batch_loss = (loss / int(target.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    # 计算梯度
    gradients = tape.gradient(loss, variables)
    # 使用优化器优化参数值，进行拟合
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss
