import tensorflow as tf
import numpy as np


class Settings(object):
    def __init__(self):
        self.model_name = 'bi_lstm_crf'
        self.embedding_size = 64
        self.time_step = 150  # 句子应该多长
        self.hidden_size = 64  # hidden size应该比输入的embedding size大吧
        self.layers_num = 2  # 增大的话，会多大程度影响性能
        self.n_classes = 31  # UNK标注成什么？还有不足time step进行补位的项目
        self.vocabulary_size = 3000
        self.ckpt_path = r'C:\Users\nhn\Documents\GitHub\ml_learn\tensorflow_with_pycharm\nlp_competition\ckpt\\'\
                         + self.model_name + '/'
        self.summary_path = ''


class BiLstmCRF(object):
    def __init__(self, settings):
        self.embedding_size = settings.embedding_size
        self.time_step = settings.time_step
        self.hidden_size = settings.hidden_size
        self.layers_num = settings.layers_num
        self.n_classes = settings.n_classes
        self.vocabulary_size = settings.vocabulary_size

        self._keep_prob = tf.placeholder(tf.float32, [])
        # input placeholder
        with tf.name_scope('Inputs'):
            self._x_inputs = tf.placeholder(tf.int64, [None, self.time_step], name='x_input')
            self._y_inputs = tf.placeholder(tf.int64, [None, self.time_step], name='y_input')
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(shape=[self.vocabulary_size+1, self.embedding_size],
                                             initializer=tf.random_uniform_initializer,
                                             dtype=tf.float32, trainable=True, name='embedding')
        with tf.variable_scope('bi_lstm'):
            self.test_output = self.inference(self.x_inputs)

    @property
    def x_inputs(self):
        return self._x_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

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
        return inputs


if __name__ == '__main__':
    my_settings = Settings()
    network = BiLstmCRF(my_settings)
    data = np.load(r'C:\Users\nhn\Desktop\data\process_data\0.npz')
    # print(data['X'].shape)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_result = network.test_output.eval(feed_dict={network.x_inputs: data['X']})
        print(test_result)

