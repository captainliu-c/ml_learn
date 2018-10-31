import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import network


flags = tf.flags
flags.DEFINE_float('lr', 1e-3, 'initial learning rate, default: 1e-3')
flags.DEFINE_float('decay_rate', 0.60, 'lr decay rate, default: 0.65')
flags.DEFINE_integer('decay_step', 207, 'decay_step, default: 180')
flags.DEFINE_integer('valid_step', 140, 'valid_step, default: 10000')  # 132=(220*6)/10
flags.DEFINE_integer('max_epoch', 8, 'all training epochs, default: 6')
# flags.DEFINE_float('last_f1', 0.40, 'if valid_f1 > last_f1, save new model. default: 0.40')
flags.DEFINE_float('last_accuracy', 0.90, 'if valid_accuracy > last_accuracy, save new model. default: 0.90')
flags.DEFINE_bool('submit_control', True, 'if True, the train.py will use cpk to predict submit data')

SETTINGS = network.Settings()
ROOT_PATH = SETTINGS.root_path
DATA_TRAIN_PATH = ROOT_PATH + r'\data\process_data\train\\'
DATA_VALID_PATH = ROOT_PATH + r'\data\process_data\validation\\'
DATA_SUBMIT_PATH = ROOT_PATH + r'\data\process_data\result\\'


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
LAST_ACCURACY = FLAGS.last_accuracy
MODEL_PATH = SETTINGS.ckpt_path + 'model.ckpt'
SUMMARY_PATH = SETTINGS.summary_path
# TRAIN_BATCH_SIZE = VALID_BATCH_SIZE = SETTINGS.train_and_validation_size
SUBMIT_CONTROL = FLAGS.submit_control


def get_batch(data_path, data_id):
    data = np.load(data_path+str(data_id)+'.npz')
    return data['X'], data['y'], data['len']


def valid_epoch(path, model, sess):
    _accuracy_list = list()
    for batch_id in range(N_VA_BATCHES):
        x_batch, y_batch, sentence_length = get_batch(path, batch_id)  # fetches = [model.accuracy]
        feed_dict = {model.x_inputs: x_batch, model.y_inputs: y_batch, model.batch_size: 100,
                     model.sentence_lengths: sentence_length, model.dropout_prob: NOT_TRAIN_DROPOUT}
        accuracy = sess.run(model.accuracy, feed_dict)
        _accuracy_list.append(accuracy)
    mean_accuracy = sess.run(tf.reduce_mean(_accuracy_list))
    return mean_accuracy


def train_epoch(path, model, sess, train_fetches, valid_fetches, train_writer, test_writer):
    global LEARNING_RATE
    global LAST_ACCURACY

    for batch in tqdm(range(N_TR_BATCHES)):
        global_steps = sess.run(model.global_steps)
        if (global_steps+1) % FLAGS.valid_step == 0:
            mean_accuracy = valid_epoch(path, model, sess)
            if mean_accuracy > LAST_ACCURACY:
                LAST_ACCURACY = mean_accuracy
                saving_path = model.saver.save(sess, MODEL_PATH, global_steps+1)
                print('saved new model to %s ' % saving_path)
            print('the step is %d and the validation mean accuracy is %g' % (global_steps + 1, mean_accuracy))

        random_batch = np.random.permutation(N_TR_BATCHES)
        batch = random_batch[batch]
        x_data, y_data, sentence_lengths = get_batch(DATA_TRAIN_PATH, batch)
        feed_dict = {model.x_inputs: x_data, model.y_inputs: y_data, model.batch_size: 100,
                     model.dropout_prob: TRAIN_DROPOUT, model.sentence_lengths: sentence_lengths}
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

    with tf.Session(config=config) as sess:
        model = network.BiLstmCRF(SETTINGS)
        with tf.variable_scope('training_op') as vs:
            learning_rate = tf.train.exponential_decay(LEARNING_RATE, model.global_steps,
                                                       DECAY_STEP, LR_DECAY_RATE, staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(model.lost, model.global_steps)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(SUMMARY_PATH + 'train', sess.graph)
            test_writer = tf.summary.FileWriter(SUMMARY_PATH + 'test')
            training_ops = [v for v in tf.global_variables() if v.name.startswith(vs.name+'/')]

        if os.path.exists(SETTINGS.ckpt_path + 'checkpoint'):
            print('Restoring Variables from Checkpoint...')
            model.saver.restore(sess, tf.train.latest_checkpoint(SETTINGS.ckpt_path))
            # mean_accuracy = valid_epoch(DATA_TRAIN_PATH, model, sess)
            # print('valid mean accuracy is %g' % mean_accuracy)
            sess.run(tf.variables_initializer(training_ops))
        else:
            print('Initializing Variables...')
            tf.global_variables_initializer().run()

        if SUBMIT_CONTROL:
            # SETTINGS.batch_size = 211
            total_predict, total_belong = [], []
            # 输入：x_batch、sentence_length， 获得预测结果 model.predict_sentence
            for batch_id in tqdm(range(24)):
                submit_data = np.load(DATA_SUBMIT_PATH + str(batch_id) + '.npz')
                submit_x, submit_length, submit_belong = submit_data['X'], submit_data['len'], submit_data['belong']
                feed_dict = {model.x_inputs: submit_x, model.sentence_lengths: submit_length,
                             model.batch_size: 211, model.dropout_prob: NOT_TRAIN_DROPOUT}
                _predict = sess.run(model.predict_sentence, feed_dict=feed_dict)
                total_predict.append(_predict)
                total_belong.append(submit_belong)
            # 读取total_predict和total_belong，生成ann
            # 获得实体类别、两端位置，对应的raw data切片 | 创建两个集合，一个是begin集合，一个是end集合
            # 检查符合规则的实体 |
                # 检查开始：针对每一行sentence的字，字是否出现在开始集合中。并且所处位置的索引的前一项和后一项不能是任意开头索引
                # 检查结束：从查找索引的位置开始，以连续两个0出现的位置为终点
                # 将0替换为；来获得最终的位置情况
                # 获得开始和中止位置，打开对应belong的txt，对txt进行写入。写入格式为：T%d  根据开头的实体类别 位置情况 raw_x的切片
                break
        else:
            for epoch in range(FLAGS.max_epoch):
                global_step = sess.run(model.global_steps)
                print('the step is %d, and the learning rate is %g' % (global_step, sess.run(learning_rate)))
                train_fetches = [merged, train_op]
                valid_fetches = [merged, model.lost]
                train_epoch(DATA_VALID_PATH, model, sess, train_fetches, valid_fetches, train_writer, test_writer)


if __name__ == '__main__':
    main()
