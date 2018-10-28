import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode


class Settings(object):
    def __init__(self):
        self.model_name = 'bi_lstm_crf'
        self.embedding_size = 64
        self.time_step = 150  # 句子应该多长
        self.hidden_size = 64  # hidden size应该比输入的embedding size大吧
        self.layers_num = 2  # 增大的话，会多大程度影响性能
        self.n_classes = 31  # UNK标注成什么？还有不足time step进行补位的项目
        self.vocabulary_size = 3000
        self.weights_decay = 0.001  # 按照参数传进来
        self.batch_size = 128
        self.ckpt_path = r'C:\Users\nhn\Documents\GitHub\ml_learn\tensorflow_with_pycharm\nlp_competition\ckpt\\'\
                         + self.model_name + '/'
        self.summary_path = ''


class BiLstmCRF(object):
    """
    bi_lstm[+dropout] ->flatten->crf
    """
    def __init__(self, settings):
        self.embedding_size = settings.embedding_size
        self.time_step = settings.time_step
        self.hidden_size = settings.hidden_size
        self.layers_num = settings.layers_num
        self.n_classes = settings.n_classes
        self.vocabulary_size = settings.vocabulary_size
        self._weights_decay = settings.weights_decay
        self._batch_size = settings.batch_size
        self._global_steps = tf.Variable(0, trainable=False, name='Global_Step')

        self._dropout_prob = tf.placeholder(tf.float32, [])
        # input placeholder
        with tf.name_scope('Inputs'):
            self._x_inputs = tf.placeholder(tf.int64, [None, self.time_step], name='x_input')
            self._y_inputs = tf.placeholder(tf.int64, [None, self.time_step], name='y_input')
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(shape=[self.vocabulary_size+1, self.embedding_size],
                                             initializer=tf.random_uniform_initializer,
                                             dtype=tf.float32, trainable=True, name='embedding')
        with tf.variable_scope('bi_lstm'):
            bi_lstm_output = self.inference(self.x_inputs)
            bi_lstm_output = tf.nn.dropout(bi_lstm_output, self._dropout_prob)
        with tf.variable_scope('flatten'):
            flatten_input = tf.reshape(bi_lstm_output, [-1, self.hidden_size * 2])
            weights = self._variable_with_weight_decay('weights', [self.hidden_size*2, self.n_classes],
                                                       0.1, self.weights_decay)
            biases = self._variable_on_cpu('biases', [self.n_classes], tf.constant_initializer(0.1))
            flatten_out = tf.matmul(flatten_input, weights)+biases
        with tf.name_scope('crf'):  # 没用variable_scope
            self.logits = tf.reshape(flatten_out, [-1, self.time_step, self.n_classes])
            self.sequence_lengths = np.full(self.batch_size, self.time_step, dtype=np.int32)
            # 需要改data_process，把句子的真实长度保存到npz中，用length做索引
            log_likelihood, self.transition_params = crf_log_likelihood(
                inputs=self.logits, tag_indices=self.y_inputs, sequence_lengths=self.sequence_lengths)
            self.crf_loss = tf.reduce_mean(log_likelihood)
            self.lost = self.crf_loss + tf.add_n(tf.get_collection('losses'))
        self.saver = tf.train.Saver(max_to_keep=2)

    @property
    def x_inputs(self):
        return self._x_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    @property
    def weights_decay(self):
        return self._weights_decay

    @property
    def dropout_prob(self):
        return self._dropout_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_steps(self):
        return self._global_steps

    @staticmethod
    def _variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wb):
        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))

        if wb:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wb)
            tf.add_to_collection('losses', weight_decay)
        return var

    def inference(self, x_inputs):
        inputs = tf.nn.embedding_lookup(self.embedding, x_inputs)
        cell_fw = LSTMCell(self.hidden_size)
        cell_bw = LSTMCell(self.hidden_size)
        (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs, dtype=tf.float32)
        outputs = tf.concat([outputs_fw, outputs_bw], axis=-1)
        return outputs


if __name__ == '__main__':
    my_path = r'C:\Users\Neo\Documents\GitHub\ml_learn\tensorflow_with_pycharm' \
              r'\nlp_competition\data\process_data\train\0.npz'
    data = np.load(my_path)
    my_settings = Settings()
    network = BiLstmCRF(my_settings)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        feed_dict = {network.x_inputs: data['X'], network.y_inputs: data['y'], network.dropout_prob: 1}
        print(network.lost.eval(feed_dict=feed_dict))
