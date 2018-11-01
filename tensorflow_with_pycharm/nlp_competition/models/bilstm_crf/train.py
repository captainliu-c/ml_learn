import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import network
from data_process import DataProcess
from tools import flatten

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
DATA_SUBMIT_OUT_PATH = ROOT_PATH + r'\data\process_data\submit\\'

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
            _data_process = DataProcess()
            total_predict, total_belong = [], []
            file2batch_relationship_reverse = _data_process.file2batch_relationship_reverse
            entity_tag = _data_process.tags
            begin_tag_index = [x for x in list(entity_tag.values())[1:] if (x+1) % 2 == 0]

            # 输入：x_batch、sentence_length， 获得预测结果 model.predict_sentence
            for batch_id in tqdm(range(24)):
                submit_data = np.load(DATA_SUBMIT_PATH + str(batch_id) + '.npz')
                submit_x, submit_length, submit_belong = submit_data['X'], submit_data['len'], submit_data['belong']
                feed_dict = {model.x_inputs: submit_x, model.sentence_lengths: submit_length,
                             model.batch_size: 211, model.dropout_prob: NOT_TRAIN_DROPOUT}
                _predict = sess.run(model.predict_sentence, feed_dict=feed_dict)
                batch_index = 0
                while batch_index < len(_predict):
                    sentence, sentence_length = _predict[batch_index], submit_length[batch_index]
                    total_predict.append(sentence[:sentence_length])  # 恢复为真实长度
                    batch_index += 1
                total_belong.extend(submit_belong)
            # 添加句号，恢复原始index
            _files, files = [], []  # 索引即是ID
            start_index = 0
            for file_id in range(59):  # 还原成真实文件对应的句子数量
                file_size = total_belong.count(file_id)
                _files.append(total_predict[start_index:start_index+file_size])
                start_index = start_index+file_size
            for file in _files:  # 添加句号，处理成二维数组
                sentence_index = 1
                flatten_article = list(file[0])
                while sentence_index < len(file):
                    flatten_article.extend('。')
                    flatten_article.extend(file[sentence_index])
                    sentence_index += 1
                files.append(flatten_article)
            print(len(files[0]))
            # 检查文件[127——1.txt]长度不一致的问题，原文件3017，恢复后2169多

            # for file in files:
            #     # 获得文件名
            #     file_name = file2batch_relationship_reverse[files.index(file)]  # file在files中是唯一的
            #     global_id = 1
            #     file_index = 0
            #     # 查找合法的实体开头
            #     while file_index < len(file)-1:
            #         word, next_word = file[file_index], file[file_index+1]
            #         if word in begin_tag_index and (next_word == word+1):
            #             print('i have found the beginning of the entity')
            #             entity_begin_index = file_index
            #             # 查找实体结尾
            #             while True:
            #                 entity_index = file_index+1
            #                 entity_body, next_entity_body = file[entity_index], file[entity_index+1]
            #                 if entity_body != 0 and next_entity_body != 0:
            #                     entity_index += 1
            #                 else:
            #                     entity_end_index = entity_index
            #                     file_index = entity_end_index-1  # 末尾file_index+1
            #                     break
            #             print('the begin and end index of the entity are: %d | %d'
            #                   % (entity_begin_index, entity_end_index))
            #             # 获得实体类别
            #             entity_type = entity_tag.get(file[entity_begin_index], 'wrong')
            #             if entity_type == 'wrong':
            #                 raise ValueError('could not find the key:', file[entity_begin_index])
            #             # 将global_id、实体类别、index相关、empty写入文件
            #             content_1 = 'T' + str(global_id) + '\t'
            #             content_3 = '\t' + 'empty'
            #             # 生成实体的index，处理换行的情况
            #             index_pair = [entity_begin_index]
            #             entity_slice = file[entity_begin_index:entity_end_index]
            #             for j in entity_slice:
            #                 if j == 0:
            #                     _new_begin = entity_begin_index+entity_slice.index(j)
            #                     index_pair.append(_new_begin)
            #                     index_pair.append(_new_begin+1)
            #             index_pair.append(entity_end_index)
            #             # 补全content_2
            #             content_2 = entity_type + ' ' + str(index_pair[0])
            #             final_index = 1
            #             for _ in range(int(len(index_pair)/2)-1):
            #                 content_2 = (content_2 + ' '
            #                              + str(index_pair[final_index]) + ';' + str(index_pair[final_index+1]))
            #                 final_index += 2
            #             content_2 = content_2 + ' ' + str(index_pair[-1])
            #             # 合并content
            #             content = content_1+content_2+content_3+'\n'
            #             global_id += 1
            #             # 写入文件
            #             submit_path = DATA_SUBMIT_OUT_PATH + file_name.replace('txt', 'ann')
            #             with open(submit_path, 'a') as f:  # 如果file不存在，会创建吗
            #                 f.write(content)
            #         file_index += 1

        else:
            for epoch in range(FLAGS.max_epoch):
                global_step = sess.run(model.global_steps)
                print('the step is %d, and the learning rate is %g' % (global_step, sess.run(learning_rate)))
                train_fetches = [merged, train_op]
                valid_fetches = [merged, model.lost]
                train_epoch(DATA_VALID_PATH, model, sess, train_fetches, valid_fetches, train_writer, test_writer)


if __name__ == '__main__':
    main()
