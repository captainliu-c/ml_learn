import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import network
from tensorflow.contrib.crf import crf_decode
# 周一需要同时再改一下[除了保存原本句子长度]data process, 追加一下shuffle和区分validation


flags = tf.flags
flags.DEFINE_float('lr', 1e-3, 'initial learning rate, default: 1e-3')
flags.DEFINE_float('decay_rate', 0.65, 'lr decay rate, default: 0.65')
flags.DEFINE_integer('decay_step', 220, 'decay_step, default: 180')  # train batch的数量
flags.DEFINE_integer('valid_step', 48, 'valid_step, default: 10000')  # 看看实际有多少
flags.DEFINE_integer('max_epoch', 6, 'all training epoches, default: 6')
flags.DEFINE_float('last_f1', 0.40, 'if valid_f1 > last_f1, save new model. default: 0.40')

SETTINGS = network.Settings()
DATA_TRAIN_PATH = r'C:\Users\nhn\Documents\GitHub\ml_learn\tensorflow_with_pycharm\nlp_competition\data\process_data\train\\'
DATA_VALID_PATH = r'C:\Users\nhn\Documents\GitHub\ml_learn\tensorflow_with_pycharm\nlp_competition\data\process_data\validation\\'
TR_BATCHES = os.listdir(DATA_TRAIN_PATH)  # 会变吗 写在函数里
VA_BATCHES = os.listdir(DATA_VALID_PATH)
N_TR_BATCHES = len(TR_BATCHES)
N_VA_BATCHES = len(VA_BATCHES)
FLAGS = flags.FLAGS
LEARNING_RATE = FLAGS.lr
LR_DECAY_RATE = FLAGS.decay_rate
DECAY_STEP = FLAGS.decay_step
TRAIN_DROPOUT = 0.5
NOT_TRAIN_DROPOUT = 1.0


def get_batch(data_path, data_id):
    data = np.load(data_path+str(data_id)+'.npz')
    return data['X'], data['y'], data['len']


def valid_epoch(path, model, sess):
    _batches = os.listdir(path)
    _n_batches = len(_batches)
    for i in range(_n_batches):
        x_batch, y_batch, sentence_length = get_batch(path, i)
        fetches = [model.transition_params]
        feed_dict = {model.x_inputs: x_batch, model.y_inputs: y_batch, model.dropout_prob: NOT_TRAIN_DROPOUT}
# def valid_epoch(path, model, sess):
#     _batches = os.listdir(path)
#     _n_batches = len(_batches)
#     cost = 0.0
#     # for i in range(_n_batches):
#     for i in range(180, 209):
#         x_batch, y_batch = get_batch(path, i)
#         fetches = [model.lost]
#         feed_dict = {model.x_inputs: x_batch, model.y_inputs: y_batch, model.dropout_prob: NOT_TRAIN_DROPOUT}
#         _cost = sess.run(fetches, feed_dict)[0]  # 后续增加fetches的话，这里需要修改
#         cost += _cost
#     mean_cost = cost/_n_batches
#     return mean_cost


def train_epoch(path, model, sess, train_fetches):
    global LEARNING_RATE

    for batch in tqdm(range(N_TR_BATCHES)):
        random_batch = np.random.permutation(N_TR_BATCHES)
        batch = random_batch[batch]
        x_data, y_data, sentence_lengths = get_batch(DATA_TRAIN_PATH, batch)
        feed_dict = {model.x_inputs: x_data, model.y_inputs: y_data,
                     model.dropout_prob: TRAIN_DROPOUT, model.sequence_lengths: sentence_lengths}
        global_steps = sess.run(model.global_steps)

        if (global_steps+1) % 10 == 0:
            lost = sess.run(model.lost, feed_dict)
            print('the step is %d and the train data set lost is %g' % (global_steps+1, lost))
        #     lost = valid_epoch(path, model, sess)
        #     print('the step is %d and the lost is %g' % (global_steps+1, lost))
        sess.run(train_fetches, feed_dict)


def main():
    config = tf.ConfigProto(
        device_count={'CPU': 4},
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4)

    with tf.Session(config=config) as sess:
        model = network.BiLstmCRF(SETTINGS)
        with tf.variable_scope('training_op'):
            learning_rate = tf.train.exponential_decay(LEARNING_RATE, model.global_steps,
                                                       DECAY_STEP, LR_DECAY_RATE, staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(model.lost, model.global_steps)
        tf.global_variables_initializer().run()
        for epoch in range(FLAGS.max_epoch):
            global_step = sess.run(model.global_steps)
            train_fetches = [train_op]
            train_epoch(DATA_VALID_PATH, model, sess, train_fetches)
            print('the step is %d, and the learning rate is %g' % (global_step, sess.run(learning_rate)))
            break


if __name__ == '__main__':
    main()
