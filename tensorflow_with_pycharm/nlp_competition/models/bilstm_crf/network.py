import tensorflow as tf
import os
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import crf_decode
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers


class Settings(object):
    def __init__(self):
        self.model_name = 'bi_lstm_crf'
        self.embedding_size = 50  # 即使500的话，效果也不好
        self.hidden_size = 50  # 即使500的话，效果也不好
        self.layers_num = 1
        self.embed_dropout_prob = 1.0
        self.seq_dim = 40

        self.weights_decay = 0.001
        self.time_step = 150
        self.n_classes = 31
        self.n_seq = 4
        self.vocabulary_size = 3000
        self.root_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")))
        self.ckpt_path = self.root_path + r'/ckpt/' + self.model_name + '/'
        self.summary_path = self.root_path + r'/summary/' + self.model_name + '/'


class BiLstmCRF(object):
    """
    bi_lstm[+dropout] ->flatten[hidden_size*2, hidden_size]->tanh->flatten[hidden_size, n_classes]->crf
    """
    def __init__(self, settings):
        self.embedding_size = settings.embedding_size
        self.time_step = settings.time_step
        self.hidden_size = settings.hidden_size
        self.seq_dim = settings.seq_dim
        self.layers_num = settings.layers_num
        self.n_classes = settings.n_classes
        self.n_seq = settings.n_seq
        self.vocabulary_size = settings.vocabulary_size
        self._weights_decay = settings.weights_decay
        self._embed_dropout_prob = settings.embed_dropout_prob
        self._global_steps = tf.Variable(0, trainable=False, name='Global_Step')
        self.initializer = initializers.xavier_initializer()

        self._dropout_prob = tf.placeholder(tf.float32, [])
        # input placeholder
        with tf.name_scope('Inputs'):
            self._sentence_lengths = tf.placeholder(tf.int32, [None], name='sentence_lengths')
            self._x_inputs = tf.placeholder(tf.int32, [None, self.time_step], name='x_input')
            self._y_inputs = tf.placeholder(tf.int32, [None, self.time_step], name='y_input')
            self._seq_inputs = tf.placeholder(tf.int32, [None, self.time_step], name='seq_input')
            self._batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        self._embedding = []
        with tf.variable_scope('char_embedding'):
            self.char_embedding = tf.get_variable(shape=[self.vocabulary_size+1, self.embedding_size],
                                                  initializer=self.initializer,
                                                  dtype=tf.float32, trainable=True, name='char_embedding')
            self._embedding.append(tf.nn.embedding_lookup(self.char_embedding, self.x_inputs))
            with tf.variable_scope('seq_embedding'):
                self.seq_embedding = tf.get_variable(shape=[self.n_seq, self.seq_dim],
                                                     initializer=self.initializer,
                                                     dtype=tf.float32, trainable=True, name='seq_embedding')
                self._embedding.append(tf.nn.embedding_lookup(self.seq_embedding, self.seq_inputs))
            self.embedding = tf.concat(self._embedding, axis=-1)
            self.embedding = tf.nn.dropout(self.embedding, self._embed_dropout_prob)
        with tf.variable_scope('bi_lstm'):
            bi_lstm_output = self.inference(self.embedding)
            bi_lstm_output = tf.nn.dropout(bi_lstm_output, self._dropout_prob)
        with tf.variable_scope('flatten_middle'):
            flatten_input = tf.reshape(bi_lstm_output, [-1, self.hidden_size * 2])
            weights = self._variable_with_weight_decay('weights_middle', [self.hidden_size*2, self.hidden_size],
                                                       self.weights_decay)
            tf.summary.histogram('weights_middle', weights)
            biases = self._variable_on_cpu('biases_middle', [self.hidden_size], tf.zeros_initializer())
            tf.summary.histogram('biases_middle', biases)
            _flatten_middle = tf.matmul(flatten_input, weights)+biases
            flatten_middle = tf.tanh(_flatten_middle)
        with tf.variable_scope('flatten_out'):
            weights = self._variable_with_weight_decay('weights_out', [self.hidden_size, self.n_classes],
                                                       self.weights_decay)
            tf.summary.histogram('weights_out', weights)
            biases = self._variable_on_cpu('biases_out', [self.n_classes], tf.zeros_initializer())
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
    def seq_inputs(self):
        return self._seq_inputs

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

    def inference(self, inputs):
        def basic_cell(hidden_size, initializer):
            _cell = rnn.CoupledInputForgetGateLSTMCell(
                hidden_size,
                use_peepholes=True,
                initializer=initializer)
            return _cell

        lstm_cell = {}
        for direction in ['forward', 'backward']:
            with tf.variable_scope(direction):
                lstm_cell[direction] = rnn.MultiRNNCell(
                    [basic_cell(self.hidden_size, self.initializer) for _ in range(self.layers_num)])
        (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell['forward'], cell_bw=lstm_cell['backward'], inputs=inputs,
            sequence_length=self.sentence_lengths, dtype=tf.float32)
        outputs = tf.concat([outputs_fw, outputs_bw], axis=-1)

        return outputs


if __name__ == '__main__':
    pass
