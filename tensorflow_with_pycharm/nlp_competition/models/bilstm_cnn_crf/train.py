import tensorflow as tf
import numpy as np
import network
import os
from tqdm import tqdm
from bilstm_cnn_crf.tools import get_batch

SETTINGS = network.Settings()
ROOT_PATH = SETTINGS.root_path
DATA_TRAIN_PATH = ROOT_PATH + '/data/process_data/train/'
DATA_VALID_PATH = ROOT_PATH + '/data/process_data/validation/'
MODEL_PATH = SETTINGS.ckpt_path + 'model.ckpt'
SUMMARY_PATH = SETTINGS.summary_path
TRAIN_DROPOUT, NOT_TRAIN_DROPOUT = 0.5, 1.0
TRAIN_BATCH_SIZE = 200
N_TR_BATCHES, N_VA_BATCHES = len(os.listdir(DATA_TRAIN_PATH)), len(os.listdir(DATA_VALID_PATH))

LEARNING_RATE = 1e-3
LR_DECAY_RATE = 0.60
DECAY_STEP = 154
LAST_ACCURACY = 0.90
CLIP = 5
MAX_EPOCH = 20
VALID_STEP = 150


def get_feed_dict(model, get_batch_return, batch_size, feed_control):
    """
    :param model: network
    :param get_batch_return: train or valid --> x, y, sentence_length, inword : 4
                                     submit --> x, len, belong, comma, inword : 5
    :param feed_control: is_train, is_valid, is_submit
    :param batch_size: train[valid]batch_size or submit batch_size
    :return: feed_dict
    """
    data_list = list(get_batch_return)
    feed_dict = {model.x_inputs: data_list[0], model.seq_inputs: data_list[1], model.sentence_lengths: data_list[2],
                 model.batch_size: batch_size}
    if feed_control in ['is_train', 'is_valid']:
        feed_dict[model.y_inputs] = data_list[3]
        feed_dict[model.dropout_prob] = TRAIN_DROPOUT
        if feed_control == 'is_valid':
            feed_dict[model.dropout_prob] = NOT_TRAIN_DROPOUT
    elif feed_control == 'is_submit':
        feed_dict[model.dropout_prob] = NOT_TRAIN_DROPOUT
    else:
        raise ValueError('no control type:', feed_control)
    return feed_dict


def valid_epoch(path, model, sess):
    _accuracy_list = list()
    for batch_id in range(N_VA_BATCHES):
        feed_dict = get_feed_dict(model, get_batch(path, batch_id), TRAIN_BATCH_SIZE, feed_control='is_valid')
        accuracy = sess.run(model.accuracy, feed_dict)
        _accuracy_list.append(accuracy)
    mean_accuracy = sess.run(tf.reduce_mean(_accuracy_list))
    return mean_accuracy


def train_epoch(valid_path, model, sess, train_fetches, valid_fetches, train_writer, test_writer):
    global LEARNING_RATE
    global LAST_ACCURACY

    random_batch = np.random.permutation(N_TR_BATCHES)
    for batch in tqdm(range(N_TR_BATCHES)):
        global_steps = sess.run(model.global_steps)
        if (global_steps+1) % VALID_STEP == 0:
            mean_accuracy = valid_epoch(valid_path, model, sess)
            if mean_accuracy > LAST_ACCURACY:
                LAST_ACCURACY = mean_accuracy
                saving_path = model.saver.save(sess, MODEL_PATH, global_steps+1)
                print('saved new model to %s ' % saving_path)
            print('the step is %d and the validation mean accuracy is %g' % (global_steps + 1, mean_accuracy))

        batch_id = random_batch[batch]
        feed_dict = get_feed_dict(model, get_batch(DATA_TRAIN_PATH, batch_id),
                                  TRAIN_BATCH_SIZE, feed_control='is_train')
        summary, _ = sess.run(train_fetches, feed_dict)

        if (global_steps+1) % 20 == 0:
            train_writer.add_summary(summary, global_steps)
            summary, lost = sess.run(valid_fetches, feed_dict)
            test_writer.add_summary(summary, global_steps)
            print('the step is %d and the train data set lost is %g' % (global_steps+1, lost))


def main():
    config = tf.ConfigProto(
        device_count={'CPU': 4},
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = network.BiLstmCNNCRF(SETTINGS)
        with tf.variable_scope('training_op') as vs:
            learning_rate = tf.train.exponential_decay(LEARNING_RATE, model.global_steps,
                                                       DECAY_STEP, LR_DECAY_RATE, staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.lost)
            grads_and_vars_clips = [[tf.clip_by_value(g, -CLIP, CLIP), v] for g, v in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars_clips, global_step=model.global_steps)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(SUMMARY_PATH + 'train', sess.graph)
            test_writer = tf.summary.FileWriter(SUMMARY_PATH + 'test')
            training_ops = [v for v in tf.global_variables() if v.name.startswith(vs.name+'/')]

        if os.path.exists(SETTINGS.ckpt_path + 'checkpoint'):
            print('Restoring Variables from Checkpoint...')
            model.saver.restore(sess, tf.train.latest_checkpoint(SETTINGS.ckpt_path))
            mean_accuracy = valid_epoch(DATA_VALID_PATH, model, sess)
            print('valid mean accuracy is %g' % mean_accuracy)
            sess.run(tf.variables_initializer(training_ops))
        else:
            print('Initializing Variables...')
            tf.global_variables_initializer().run()

        for epoch in range(MAX_EPOCH):
            global_step = sess.run(model.global_steps)
            print('the step is %d, and the learning rate is %g' % (global_step, sess.run(learning_rate)))
            train_fetches = [merged, train_op]
            valid_fetches = [merged, model.lost]
            train_epoch(DATA_VALID_PATH, model, sess, train_fetches, valid_fetches, train_writer, test_writer)


if __name__ == '__main__':
    main()
