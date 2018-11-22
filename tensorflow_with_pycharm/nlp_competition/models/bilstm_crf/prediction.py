import tensorflow as tf
import numpy as np
import os
import network
from tqdm import tqdm
from bilstm_cnn_crf.tools import get_batch
from data_process import DataProcess
from train import valid_epoch, get_feed_dict

SETTINGS = network.Settings()
SUBMIT_BATCH_SIZE = 129

ROOT_PATH = SETTINGS.root_path
DATA_VALID_PATH = ROOT_PATH + r'/data/process_data/validation/'
DATA_SUBMIT_PATH = ROOT_PATH + r'/data/process_data/result/'
DATA_SUBMIT_OUT_PATH = ROOT_PATH + r'/data/process_data/submit/'
N_SUB_BATCHES = len(os.listdir(DATA_SUBMIT_PATH))
N_VA_BATCHES = len(os.listdir(DATA_VALID_PATH))
middle_files_path = ROOT_PATH + r'/middle/files.npy'  # 只是给这一个文件用的


def predict(model, sess):
    _data_process = DataProcess()
    total_predict, total_belong, is_commas_total = [], [], []
    file2batch_relationship_reverse = _data_process.file2batch_relationship_reverse
    _entity_tag = _data_process.tags
    entity_tag = dict(zip(_entity_tag.values(), _entity_tag.keys()))
    begin_tag_index = [x for x in list(_entity_tag.values())[1:] if (x + 1) % 2 == 0]

    if os.path.exists(middle_files_path):
        files = list(np.load(middle_files_path))
    else:
        # 获得batch的预测结果
        for batch_id in tqdm(range(N_SUB_BATCHES)):  # 24=batch数量
            get_batch_return = get_batch(DATA_SUBMIT_PATH, batch_id, is_submit=True)
            # submit_x, submit_length, submit_belong, is_comma, submit_inword = get_batch_return
            _, _, submit_length, submit_belong, is_comma = get_batch_return
            feed_dict = get_feed_dict(model, get_batch_return, SUBMIT_BATCH_SIZE, feed_control='is_submit')
            _predict = sess.run(model.predict_sentence, feed_dict=feed_dict)
            # 恢复sentence为真实长度
            __index = 0
            for pred_sentence in _predict:
                len_sentence = list(submit_length)[__index]  # predict和length在源数据中必须一一对应
                total_predict.append(pred_sentence[:len_sentence])
                __index += 1
            is_commas_total.extend(is_comma)
            total_belong.extend(submit_belong)
        # 添加句号，恢复原始index
        files = []  # 索引即是ID
        global_index = 0
        for file_id in range(59):  # 50=submit原文件数量
            file_size = total_belong.count(file_id)  # 确认有多少行sentence
            # print('file id: %d file size: %d' % (file_id, file_size))
            _file = total_predict[global_index:global_index + file_size]  # 切片出当前的_file

            sentence_index = 1
            flatten_article = list(_file[0])  # 将file拍成一维，并且处理句号
            while sentence_index < file_size:
                if is_commas_total[global_index + sentence_index] == 1:
                    flatten_article.extend('。')
                    flatten_article.extend(_file[sentence_index])
                else:
                    flatten_article.extend(_file[sentence_index])
                sentence_index += 1
            global_index += file_size
            files.append(flatten_article)
        np.save(middle_files_path, files)

    for file in tqdm(files):
        # 获得文件名
        file_name = file2batch_relationship_reverse[files.index(file)]  # file在files中是唯一的
        print('file name:', file_name)
        # 查找合法的实体开头
        global_id = 1
        file_index = 0
        length = len(file)
        while file_index < length - 1:
            word, next_word = file[file_index], file[file_index + 1]
            if word in begin_tag_index and (next_word == word + 1):
                entity_begin_index = file_index
                entity_head = file[entity_begin_index]
                if entity_begin_index == length - 1:  # entity开头是最后一位
                    print('i have found the entity with no body shows the last one at the file')
                    break
                # 查找实体结尾
                entity_index = entity_begin_index + 1
                if entity_index == length - 1:  # 在文档末尾这种形式 [entity_head] + [entity_body]
                    entity_end_index = entity_index
                    print('i have found 在文档末尾这种形式 [entity_head] + [entity_body]')
                else:
                    while entity_index < length - 1:
                        # print(entity_index)
                        entity_body, next_entity_body = file[entity_index], file[entity_index + 1]
                        if entity_body != entity_head + 1:
                            if entity_body == 0 and next_entity_body == entity_head + 1:
                                entity_index = min(entity_index + 1, length - 1)
                                entity_end_index = entity_index
                            else:
                                entity_end_index = entity_index
                                file_index = entity_end_index - 1
                                break
                        else:
                            entity_index = min(entity_index + 1, length - 1)
                            entity_end_index = entity_index
                # if 0 in file[entity_begin_index:entity_end_index]:
                # print('see:', file[entity_begin_index:entity_end_index])

                # # 获得实体类别
                entity_type = entity_tag.get(file[entity_begin_index], 'wrong')
                if entity_type == 'wrong':
                    raise ValueError('could not find the key:', file[entity_begin_index])
                # 将global_id、实体类别、index相关、empty写入文件
                content_1 = 'T' + str(global_id) + '\t'
                content_3 = '\t' + 'empty'
                # 生成实体的index，处理换行的情况
                index_pair = [entity_begin_index]
                entity_slice = file[entity_begin_index:entity_end_index]
                for j in entity_slice:
                    if j == 0:
                        _new_begin = entity_begin_index + entity_slice.index(j)
                        index_pair.append(_new_begin)
                        index_pair.append(_new_begin + 1)
                index_pair.append(entity_end_index)
                # 补全content_2
                content_2 = entity_type[2:] + ' ' + str(index_pair[0])
                final_index = 1
                for _ in range(int(len(index_pair) / 2) - 1):
                    content_2 = (content_2 + ' '
                                 + str(index_pair[final_index]) + ';' + str(index_pair[final_index + 1]))
                    final_index += 2
                content_2 = content_2 + ' ' + str(index_pair[-1])
                # 合并content
                content = content_1 + content_2 + content_3 + '\n'
                global_id += 1
                # 写入文件
                submit_path = DATA_SUBMIT_OUT_PATH + file_name.replace('txt', 'ann')
                with open(submit_path, 'a') as f:  # 如果file不存在，会创建吗
                    f.write(content)
            file_index += 1


def main():
    config = tf.ConfigProto(
        device_count={'CPU': 4},
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = network.BiLstmCRF(SETTINGS)
        if not os.path.exists(middle_files_path):
            print('Restoring Variables from Checkpoint...')
            model.saver.restore(sess, tf.train.latest_checkpoint(SETTINGS.ckpt_path))
            mean_accuracy = valid_epoch(DATA_VALID_PATH, model, sess)
            print('valid mean accuracy is %g' % mean_accuracy)
        predict(model, sess)


if __name__ == '__main__':
    main()
