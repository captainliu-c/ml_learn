import re
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from collections import Iterable


# 读取训练数据
def read_text(check_control='off'):
    """
    制作y
    --生成一个词列表，每一个词作为一个元素，shape=句子数量，词数量
    --构造一个将每个字转换成5-tags的函数，然后使用map将词的词列表映射为5-tags list
    --将词列表按字来合并，并截断
    """

    path = r'C:\Users\nhn\Documents\GitHub\ml_learn\tensorflow_with_pycharm\msr_training.txt'

    def check_wrap(sentences, control='off'):
        """由于编码问题，有可能出现连续的句子未换行，导致远大于32的问题"""
        length = list(map(len, sentences))
        print('the mex length of sentences is %d， the aim should be around %d'
              % (max(length), time_step))
        print('the min: ', min(length))
        print('length of the list: ', len(sentence_list))
        if control == 'on':  # 检查下奇怪的句子长度, 比如0、1、2、3、4，都是标点符号意外换行导致
            count = 0
            for i in length:
                print('the index is %d, the length is %d' % (count, i))
                if i == 10:
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

    def flatten(items, ignore_types=bytes):
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, ignore_types):
                yield from flatten(x)
            else:
                yield x

    def make_final_y(temp):
        if len(temp) > 32:
            temp = temp[:32]
        elif len(temp) < 32:
            for n in range(32-len(temp)):
                temp.append(0)
        else:
            temp = temp
        return temp

    sentence_list = re.split(r'[\n]', open(path).read())  # 列表中的一个元素是一个句子， len(s) = 86924
    if check_control == 'on':
        check_wrap(sentence_list, control='off')  # 写逻辑的话,可考虑采用：一行中如果只有符号的话，可删除
    clean_sentences = list(map(my_clean, sentence_list))  # shape = 句子数量，1
    word_list = list(map(sentence2word, clean_sentences))  # shape = 句子数量，词数量
    assert len(word_list[0][0]) != 0  # 确认是否正确拆分成二维
    result_y = []
    result_x = []
    for i in range(len(word_list)):
        temp_y = list(flatten(list(map(create_4tags, word_list[i]))))
        temp_y = make_final_y(temp_y)  # 截断、填充5th tag
        result_y.append(temp_y)

        temp_x = list(flatten(list(map(word2char, word_list[i])))) # str没法flatten
        temp_x = temp_x  # 截断, 不足的怎么办....
        result_x.append(temp_x)
    result_y = np.array(result_y)  # shape=(86909, 32)

    print(result_x[0])

    # 制作x
    # --制作字的collection
    # --将词列表合并成字列表，并截断
    # --将字列表，根据collection映射

    return result_y  # waiting result_x


# 创建超参
hidden_size = 64
layers_num = 2
time_step = 32
input_size = 128

# 创建占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, time_step], name='x')  # 经过 word embedding 映射后获得最终input
y_ = tf.placeholder(dtype=tf.float32, shape=[None, time_step], name='y_')  # shape = [batch, time_step]
keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')

# embedding = tf.Variable()  # get_variable? where?


# 对训练数据标记tag
def note_tag():
    pass


# 构建网络
def cell(hidden_size, keep_prob):
    """create basic cell"""
    basic_cell = rnn.BasicLSTMCell(num_units=hidden_size, name='basic_cell')
    drop_cell = rnn.DropoutWrapper(cell=basic_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return drop_cell


mulit_lstm = rnn.MultiRNNCell([cell(hidden_size, keep_prob) for _ in range(layers_num)])
# create forward

# create backward

# h(x), softmax

# cost func

# create session to run the net

if __name__ == '__main__':
    read_text(check_control='off')



