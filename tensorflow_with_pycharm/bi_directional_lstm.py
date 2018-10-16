import re
import logging
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from collections import Iterable
from collections import Counter

config = tf.ConfigProto(
    device_count={'CPU': 4},
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4)

train_path = r'datalab/4508/msr_training.txt'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
batch_index = 0

# 创建超参 | 如果按照one-hot来理解的话，那么数据从5138->128->64。话说正常hidden_size应该高于input的，专业才能包含更多信息
# 如果模型准高偏差的话，可以考虑将hidden_size，改为256
lr = 0.001
hidden_size = 64
layers_num = 2
time_step = 32
input_size = embedding_size = 128
vocabulary_size = 5138  # 有精力再看看怎么改
tags_classify = 5

# 创建占位符
x = tf.placeholder(dtype=tf.int32, shape=[None, time_step], name='x')  # 经过 word embedding 映射后获得最终input
y_ = tf.placeholder(dtype=tf.int32, shape=[None, time_step], name='y_')  # shape = [batch, time_step]
keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')


# 读取并处理数据
def read_text(path):
    """
    制作y
    --生成一个词列表，每一个词作为一个元素，shape=句子数量，词数量
    --构造一个将每个字转换成4-tags的函数，然后使用map将词的词列表映射为4-tags list
    --将词列表降维, 截断和添加5th tag
    制作x
    --制作字的direction
    --通过降维后的字列表，通过direction映射为数值表示, 同时对不足32的进行补位
    """

    def check_wrap(sentences, control='off'):
        """由于编码问题，有可能出现连续的句子未换行，导致远大于32的问题"""
        lengths = list(map(len, sentences))
        print('the max length of sentences is%d， the aim should be around%d' % (max(lengths), time_step))
        print('the min: ', min(lengths))
        print('length of the list: ', len(sentence_list))
        if control == 'on':
            count = 0
            for i in lengths:  # 写逻辑的话,可考虑采用：一行中如果只有符号的话，可删除
                print('the index is %d, the length is %d' % (count, i))
                if i == 10:  # 检查下奇怪的句子长度, 比如0、1、2、3、4，都是标点符号意外换行导致
                    print('==============================')
                    print('the index is %d, the length is %d' % (count, i))
                    print('count-1:', sentences[count-1])
                    print('count:', sentences[count])
                    break
                count += 1
        return None

    def my_clean(sentence):
        """去除标点符号"""
        comma = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》〔〕〚〛〜〝〞〟–—‘’‛“”„‟…‧﹏.、'
        return re.sub("[%s]+" % comma, "", sentence)

    def sentence2word(sentence):
        """transpose sentence to word, like['  我  是 大白菜 '] to ['我', '是', ‘大白菜’]"""
        assert type(sentence) == str
        result = re.split(r'\s+', sentence)
        if result[0] == '':
            result.pop(0)
        if result[-1] == '':
            result.pop(-1)
        return result

    def word2char(word):
        """transpose word to char, like [大白菜] to ['大'， '白', '菜']"""
        result = []
        for char in word:
            result.append(char)
        return result

    def create_4tags(word):
        """begin=1, middle=2, end=3, single=4"""
        tags = {'begin': 1, 'middle': 2, 'end': 3, 'single': 4}
        result = []
        word_length = len(word)
        if word_length == 0:
            raise Exception
        elif word_length == 1:
            result.append(tags['single'])
        else:
            result.append(tags['begin'])
            for j in range(word_length-2):
                result.append(tags['middle'])
            result.append(tags['end'])
        return result

    def flatten(items, ignore_types=(bytes, str)):  # ignore_types=(bytes, str)
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, ignore_types):
                yield from flatten(x)
            else:
                yield x

    def create_char_direction(sentences):
        global vocabulary_size
        unique_chars = []
        total_chars = []
        char_direction = {}
        for item in sentences:
            total_chars.extend(list(flatten(list(map(word2char, item)))))  # item是一行句子, 由N个词组成
        logger.info('total chars is done')
        unique_chars.extend(Counter(total_chars))
        logger.info('unique chars is done')
        vocabulary_size = len(unique_chars)
        # print('the size of vocabulary is %d' % vocabulary_size)
        for i in range(vocabulary_size):
            char_direction[unique_chars[i]] = i+1  # 0留给未满32位的位置
        # print('test word 人： ', char_direction['人'])
        logger.info('char direction is done')
        return char_direction

    def word2num(word):  # 接收word, 返回char在char_direction上的映射
        num_list = []
        for char in word2char(word):
            num_list.append(vocab_direction[char])
        return num_list

    with open(path) as f:
        sentence_list = re.split(r'[\n]', f.read())  # 列表中的一个元素是一个句子， len(s) = 86909
        check_wrap(sentence_list, control='off')
    clean_sentences = list(map(my_clean, sentence_list))  # shape = 句子数量，1
    word_list = list(map(sentence2word, clean_sentences))  # shape = 句子数量，词数量
    vocab_direction = create_char_direction(word_list)
    # devise_vocab = dict(zip(vocab_direction.values(), vocab_direction.keys()))
    logger.info('vocab direction is done')
    result_x, result_y = [], []
    for i in range(len(word_list)):
        temp_y = list(flatten(list(map(create_4tags, word_list[i]))))
        length = len(temp_y)
        if length > 32:
            continue  # 跳出此个循环
        elif length < 32:
            temp_x = list(flatten(list(map(word2num, word_list[i]))))
            for n in range(32-length):
                temp_y.append(0)
                temp_x.append(0)  # x 不足32怎么办？ 填0？
        result_y.append(temp_y)
        result_x.append(temp_x)
    return result_x, result_y


def make_batch(x, y, batch_num, control='off'):
    global batch_index
    test = np.arange(batch_num*time_step, dtype=int).reshape(batch_num, time_step)
    while True:
        result = []
        if batch_index > len(y):
            batch_index = 0
        if control == 'on':
            result.append(test)
        else:
            result.append(x[batch_index:batch_index+batch_num])
        result.append(y[batch_index:batch_index+batch_num])
        batch_index += batch_num
        yield result


def bi_lstm(x_inputs):
    embedding = tf.get_variable(shape=[vocabulary_size+1, 128], initializer=tf.random_uniform_initializer,
                                dtype=tf.float32, name='embedding')  # 如果有训练好的字向量更好
    inputs = tf.nn.embedding_lookup(embedding, x_inputs)

    def cell(hidden_size, keep_prob):
        """create basic cell"""
        basic_cell = rnn.BasicLSTMCell(num_units=hidden_size, name='basic_cell')
        drop_cell = rnn.DropoutWrapper(cell=basic_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        return drop_cell

    fw_mulit_lstm = rnn.MultiRNNCell([cell(hidden_size, keep_prob) for _ in range(layers_num)])
    bw_mulit_lstm = rnn.MultiRNNCell([cell(hidden_size, keep_prob) for _ in range(layers_num)])
    initial_state_fw = fw_mulit_lstm.zero_state(batch_size, dtype=tf.float32)
    initial_state_bw = bw_mulit_lstm.zero_state(batch_size, dtype=tf.float32)
    with tf.variable_scope('bi_direction'):  # 直接用tf提供的sqe2sqe应该怎么写？
        # create forward
        outputs_fw = []
        state_fw = initial_state_fw
        with tf.variable_scope('forward'):
            for i in range(time_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()  # reuse的作用域
                (output_fw, state_fw) = fw_mulit_lstm(inputs[:, i, :], state_fw)
                outputs_fw.append(output_fw)
        # create backward
        outputs_bw = []
        state_bw = initial_state_bw
        with tf.variable_scope('backward'):
            for j in range(time_step):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = bw_mulit_lstm(inputs[:, j, :], state_bw)
                outputs_bw.append(output_bw)
    # 逆转backward，合并两个结果
    outputs_bw = tf.reverse(outputs_bw, [0])
    result = tf.concat([outputs_fw, outputs_bw], 2)  # [timestep_size, batch_size, hidden_size*2]
    result = tf.transpose(result, perm=[1, 0, 2])  # [batch_size, time_step, hidden_size*2]
    result = tf.reshape(result, shape=[-1, hidden_size*2])  # [batch_size*time_step, hidden_size*2]
    return result


def h_func(lstm_result):  # h(x) with no softmax
    weight_variables = tf.Variable(tf.truncated_normal(shape=[hidden_size*2, tags_classify],
                                                       dtype=tf.float32, stddev=0.1), name='softmax_weights')
    bias_variables = tf.Variable(tf.constant(shape=[tags_classify], dtype=tf.float32, value=0.1), name='soft_bias')
    h = tf.matmul(lstm_result, weight_variables) + bias_variables
    return h


# create session to run the net
def main():
    train_batch_size = 50
    t_x, t_y = read_text(train_path)
    train_data = make_batch(t_x, t_y, train_batch_size)

    my_lstm = bi_lstm(x)
    my_h_func = h_func(my_lstm)
    # cost func
    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_, [-1]), logits=my_h_func)
    )
    train_step = tf.train.AdamOptimizer(lr).minimize(cost)
    correct_prediction = tf.equal(tf.cast(tf.argmax(my_h_func, 1), tf.int32), tf.reshape(y_, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    with tf.Session(config=config):
        tf.global_variables_initializer().run()
        for i in range(500):
            batch = next(train_data)
            train_placeholder = {x: batch[0], y_: batch[1], keep_prob: 0.5, batch_size: train_batch_size}
            train_step.run(feed_dict=train_placeholder)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict=train_placeholder)
                print('the step is %d, the accuracy is %g' % (i, train_accuracy))


if __name__ == '__main__':
    main()
