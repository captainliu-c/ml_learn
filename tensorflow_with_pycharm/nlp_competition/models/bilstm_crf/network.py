import tensorflow as tf
import numpy as np
import os
# from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import crf_decode
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers


class Settings(object):
    def __init__(self):
        self.model_name = 'bi_lstm_crf'
        self.embedding_size = 300  # 即使500的话，效果也不好
        self.time_step = 150
        self.hidden_size = 300  # 即使500的话，效果也不好
        self.layers_num = 1
        self.n_classes = 31
        self.vocabulary_size = 3000
        self.weights_decay = 0.001  # 后期再改，目前是高偏差，还没到overfit那一步
        self.root_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")))
        self.ckpt_path = self.root_path + r'\ckpt\\' + self.model_name + '\\'
        self.summary_path = self.root_path + r'\summary\\' + self.model_name + '\\'


class BiLstmCRF(object):
    """
    bi_lstm[+dropout] ->flatten[hidden_size*2, hidden_size]->tanh->flatten[hidden_size, n_classes]->crf
    """
    def __init__(self, settings):
        self.embedding_size = settings.embedding_size
        self.time_step = settings.time_step
        self.hidden_size = settings.hidden_size
        self.layers_num = settings.layers_num
        self.n_classes = settings.n_classes
        self.vocabulary_size = settings.vocabulary_size
        self._weights_decay = settings.weights_decay
        self._global_steps = tf.Variable(0, trainable=False, name='Global_Step')
        self.initializer = initializers.xavier_initializer()

        self._dropout_prob = tf.placeholder(tf.float32, [])
        # input placeholder
        with tf.name_scope('Inputs'):
            self._sentence_lengths = tf.placeholder(tf.int32, [None], name='sentence_lengths')
            self._x_inputs = tf.placeholder(tf.int32, [None, self.time_step], name='x_input')
            self._y_inputs = tf.placeholder(tf.int32, [None, self.time_step], name='y_input')
            self._batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        with tf.variable_scope('embedding'):
            self._embedding = tf.get_variable(shape=[self.vocabulary_size+1, self.embedding_size],
                                              initializer=self.initializer,
                                              dtype=tf.float32, trainable=True, name='embedding')

        with tf.variable_scope('bi_lstm'):
            bi_lstm_output = self.inference(self.x_inputs)
            bi_lstm_output = tf.nn.dropout(bi_lstm_output, self._dropout_prob)
        with tf.variable_scope('flatten_middle'):
            flatten_input = tf.reshape(bi_lstm_output, [-1, self.hidden_size * 2])
            weights = self._variable_with_weight_decay('weights_middle', [self.hidden_size*2, self.hidden_size],
                                                       self.weights_decay)
            tf.summary.histogram('weights_middle', weights)
            biases = self._variable_on_cpu('biases_middle', [self.hidden_size], tf.constant_initializer(0.1))
            tf.summary.histogram('biases_middle', biases)
            _flatten_middle = tf.matmul(flatten_input, weights)+biases
            flatten_middle = tf.tanh(_flatten_middle)
        with tf.variable_scope('flatten_out'):
            weights = self._variable_with_weight_decay('weights_out', [self.hidden_size, self.n_classes],
                                                       self.weights_decay)
            tf.summary.histogram('weights_out', weights)
            biases = self._variable_on_cpu('biases_out', [self.n_classes], tf.constant_initializer(0.1))
            tf.summary.histogram('biases_out', biases)
            flatten_out = tf.nn.xw_plus_b(flatten_middle, weights, biases)
        with tf.name_scope('crf'):  # 没用variable_scope
            self.logits = tf.reshape(flatten_out, [-1, self.time_step, self.n_classes])
            self.transition_params = tf.get_variable('transitions',
                                                     shape=[self.n_classes, self.n_classes],
                                                     initializer=self.initializer)

            log_likelihood, self.transition_params = crf_log_likelihood(
                inputs=self.logits, tag_indices=self.y_inputs,
                transition_params=self.transition_params, sequence_lengths=self.sentence_lengths)
            self._crf_loss = -tf.reduce_mean(log_likelihood)
            tf.summary.scalar('crf_lost', self._crf_loss)
            self.lost = self._crf_loss + tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('lost', self.lost)

        with tf.name_scope('predict'):
            self.predict_sentence, self.best_score = crf_decode(
                self.logits, self.transition_params, self.sentence_lengths)
            self._correct_predict = tf.equal(self.predict_sentence, self.y_inputs)
            self.accuracy = tf.reduce_mean(tf.cast(self._correct_predict, 'float'))
            tf.summary.scalar('accuracy', self.accuracy)
            # self.conf_matrix = tf.confusion_matrix(self.y_inputs, self.predict_sentence, num_classes=self.n_classes)
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

    @property
    def sentence_lengths(self):
        return self._sentence_lengths

    @staticmethod
    def _variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, wb):  # stddv
        var = self._variable_on_cpu(name, shape, self.initializer)

        if wb:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wb)
            tf.add_to_collection('losses', weight_decay)
        return var

    def inference(self, x_inputs):
        _inputs = tf.nn.embedding_lookup(self._embedding, x_inputs)
        inputs = tf.nn.dropout(_inputs, 1.0)  # self._dropout_prob
        hidden_size = self.hidden_size

        # def _cell(_hidden_size):  # no dropout
        #     basic_cell = rnn.BasicLSTMCell(num_units=_hidden_size, name='basic_cell')
        #     return basic_cell
        #
        # fw_mulit_lstm = rnn.MultiRNNCell([_cell(hidden_size) for _ in range(self.layers_num)])
        # bw_mulit_lstm = rnn.MultiRNNCell([_cell(hidden_size) for _ in range(self.layers_num)])

        lstm_cell = {}
        for direction in ['forward', 'backward']:
            with tf.variable_scope(direction):
                lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                    hidden_size,
                    use_peepholes=True,
                    initializer=self.initializer,
                    state_is_tuple=True
                )

        (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell['forward'], cell_bw=lstm_cell['backward'], inputs=inputs,
            sequence_length=self.sentence_lengths, dtype=tf.float32)

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

