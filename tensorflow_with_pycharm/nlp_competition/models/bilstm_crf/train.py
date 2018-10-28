import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import network
# 周一需要同时再改一下[除了保存原本句子长度]data process, 追加一下shuffle和区分validation


flags = tf.flags
flags.DEFINE_integer('max_epoch', 6, 'all training epoches, default: 6')
flags.DEFINE_float('lr', 1e-3, 'initial learning rate, default: 1e-3')
flags.DEFINE_integer('valid_step', 500, 'valid_step, default: 10000')  # 10000

SETTINGS = network.Settings()
DATA_TRAIN_PATH = r'C:\Users\Neo\Documents\GitHub\ml_learn' \
                  r'\tensorflow_with_pycharm\nlp_competition\data\process_data\train\\'
DATA_VALID_PATH = r'C:\Users\Neo\Documents\GitHub\ml_learn' \
                  r'\tensorflow_with_pycharm\nlp_competition\data\process_data\validation\\'
TR_BATCHES = os.listdir(DATA_TRAIN_PATH)  # 会变吗 写在函数里
VA_BATCHES = os.listdir(DATA_VALID_PATH)
N_TR_BATCHES = len(TR_BATCHES)
N_VA_BATCHES = len(VA_BATCHES)
FLAGS = flags.FLAGS
LEARNING_RATE = FLAGS.lr


def get_batch(data_path, data_id):
    data = np.load(data_path+str(data_id)+'.npz')
    return data['X'], data['y']


def valid_epoch(path, model, sess):
    _batches = os.listdir(path)
    _n_batches = len(_batches)
    cost = 0.0
    # for i in range(_n_batches):
    for i in range(170, 210):
        x_batch, y_batch = get_batch(path, i)
        fetches = [model.lost]
        feed_dict = {model.x_inputs: x_batch, model.y_inputs: y_batch, model.dropout_prob: 1.0}
        _cost = sess.run(fetches, feed_dict)
        cost += _cost
    mean_cost = cost/_n_batches
    return mean_cost


def train_epoch(path, model, sess, train_fetches):
    global LEARNING_RATE

    for batch in tqdm(range(N_TR_BATCHES)):
        random_batch = np.random.permutation(N_TR_BATCHES)
        if (model.global_steps+1) % 50 == 0:
            lost = valid_epoch(path, model, sess)
            print('the step is %d and the lost is %g' % (model.global_steps+1, lost))
        batch = random_batch[batch]
        x_data, y_data = get_batch(DATA_TRAIN_PATH, batch)
        print('the batch ID is %d' % batch)
        feed_dict = {model.x_inputs: x_data, model.y_inputs: y_data, model.dropout_prob: 0.5}
        sess.run(train_fetches, feed_dict)


def main(_):
    config = tf.ConfigProto(
        device_count={'CPU': 4},
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4)

    with tf.Session(config=config) as sess:
        model = network.BiLstmCRF(SETTINGS)
        with tf.variable_scope('training_op'):
            tvars = tf.trainable_variables()
            grads = tf.gradients(model.lost, tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_steps)
        tf.global_variables_initializer().run()
        for epoch in range(5):
            steps = sess.run(model.global_steps)
            print('the step is %d' % steps)
            train_fetches = [train_op]
            train_epoch(DATA_VALID_PATH, model, sess, train_fetches)


if __name__ == '__main__':
    tf.app.run()
