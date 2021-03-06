import jieba
import numpy as np
import os
import re
from tqdm import tqdm
from collections import Iterable, Counter
import data_process
import codecs

DICT_PATH = r'C:\Users\nhn\Documents\GitHub\ml_learn\tensorflow_with_pycharm\nlp_competition\middle'


def check_sentence_length(sentences, control=True):
    if control:
        print('--The check process is running. We will check the length of each of the sentence')
        length_sentences = list(map(len, sentences))
        index_and_length = dict(zip(length_sentences, range(len(length_sentences))))
        length_sentences.sort()
        print('---The 10 minimum length is %s ,and the index of the sentence is %s' %
              (str(length_sentences[:10]), str(list(map(index_and_length.get, length_sentences[:10])))))
        print('---The 10 maximum length is %s ,and the index of the sentence is %s' %
              (str(length_sentences[-10:]), str(list(map(index_and_length.get, length_sentences[-10:])))))
    else:
        return None


def target_delete(the_list, target=''):  # 目前只适用于一维数组
    while target in the_list:
        the_list.remove(target)
    while target in the_list:
        raise ValueError('There is a blank, the index is: ', the_list.index(target))


def flatten(the_list, skip_type=(bytes, str)):
    for item in the_list:
        if isinstance(item, Iterable) and not isinstance(item, skip_type):
            yield from flatten(item)
        else:
            yield item


def list_save_index(the_list):
    print('the length of the list is %d' % len(the_list))
    temp_list = []
    for i in the_list:
        temp_list.append(str(i+'_'+str(the_list.index(i))))
    return dict(zip(temp_list, list(range(len(the_list)))))


def check_entity_and_raw_dada(data, x_file, begin_index, end_index, check_control=True):
    if check_control:
        if len(data) > 5:  # 带换行的实体  ['T341', 'Test', 6560, 6561, 6562, 6563, '体重']
            raw_x_file = str(''.join(x_file[begin_index:end_index])+' '+''.join(x_file[data[4]:data[5]]))
            if data[-1] != raw_x_file:
                print('--the y data is: ', data)
                print('--[wrap]there is different from y | x: ', data[-1], ' | ', raw_x_file)
        else:  # 不带换行的实体 ['T346', 'Test', 6621, 6626, 'HBA1C'], 直接标注
            if data[-1] != ''.join(x_file[begin_index:end_index]):
                print('--the y data is: ', data)
                print('--there is different from y | x: ', data[-1], ' | ', ''.join(x_file[begin_index:end_index]))
            # assert data[-1] == ''.join(x_file[begin_index:end_index])


def check_final_data(x_data, y_final, times=5, gap=10, control=False):
    if control:
        i = 0
        for _ in range(times):
            print('x:', x_data[i:i+gap])
            print('y', y_final[i:i+gap])
            i += gap


def check_too_long_sentence(y_sub, time_step, control=False):
    if control:
        sentences_count = len(y_sub)
        too_long_sentence = [i for i in list(map(len, y_sub)) if i > time_step]
        print('the amount of the data[sentence] is %d' % sentences_count)
        print('the rate of the too long sentence is %g, the number is %d'
              % (len(too_long_sentence) / sentences_count, len(too_long_sentence)))
        print(too_long_sentence)


def check_sentence_wrap_and_padding(submit_final, submit_sentence_length, is_comma):
    t_index = 0
    while t_index < len(submit_final):
        print('\nthe data is: ', submit_final[t_index])
        print('the real length is %d, and the is_comma is %d' % (submit_sentence_length[t_index], is_comma[t_index]))
        print('the padding length is %d' % len(submit_final[t_index]))
        t_index += 1


def check_from_rawdata2file(raw_data_name, local_path, file):
    comma = ['。']
    raw_data_path = local_path+raw_data_name
    with open(raw_data_path, 'rb') as f:
        _data = f.read().decode('utf-8')
    raw_data = [x for x in _data]
    index = 0
    print('the file name is:', raw_data_name)
    while index < len(raw_data):
        data = raw_data[index]
        if data in comma:
            if data != file[index]:
                print('i found a different, the index is %d' % index)
            else:
                print('Same! The data is %s, and the index is %d' % (str(data), index))
        index += 1


def add_in_word_index(sentences):
    """
    0:single
    1:begin
    2:in
    3:end
    """
    sentences_with_tag = []
    for sentence in sentences:
        _cut = list(jieba.cut(''.join(sentence)))
        _sentence = []
        for word in _cut:
            if len(word) == 1:
                _sentence.append(0)
            else:
                with_tag = [2]*len(word)
                with_tag[0] = 1
                with_tag[-1] = 3
                _sentence.extend(with_tag)
        sentences_with_tag.append(_sentence)
    return sentences_with_tag


def make_oversampling(check_result, data_train_path, tags, tag_prefix, target):
    """
    原则：小于2000的样本追加到2000。原全部的数量为140028, 增加43batch，原batch102
    Disease :0.246 | count: 34413 -> top2
    Reason :0.027 | count: 3766
    Symptom :0.031 | count: 4291
    Test :0.318 | count: 44476 -> top1
    Test_Value :0.063 | count: 8776
    Drug :0.097 | count: 13541
    Frequency :0.003 | count: 485
    Amount :0.008 | count: 1078
    Method :0.006 | count: 874
    Treatment :0.007 | count: 1036
    Operation :0.004 | count: 630
    Anatomy :0.167 | count: 23392 -> top3
    Level :0.011 | count: 1539
    Duration :0.007 | count: 982
    SideEff :0.005 | count: 749
    """

    def over_sampling(name, path, number_batches):
        random_batch = np.random.permutation(number_batches)
        got_one = False
        for batch in range(number_batches):
            batch_id = random_batch[batch]

            data = np.load(path + str(batch_id) + '.npz')
            y_data = data['y']
            index = 0
            while index < len(y_data):
                sentence = y_data[index]
                if tags[tag_prefix + name] in sentence:
                    # print('got one')
                    x, y, length, inword = data['X'][index], sentence, data['len'][index], data['inword'][index]
                    got_one = True
                    break
                index += 1
            if got_one:
                break
        return x, y, length, inword

    reverse_check_result = dict(zip(check_result.values(), check_result.keys()))
    add_pool = dict()
    for entity_count in check_result.values():
        if entity_count < target:
            add_pool[reverse_check_result[entity_count]] = target-entity_count

    n_tr_batches = len(os.listdir(data_train_path))
    all_x, all_y, all_length, all_inword = [], [], [], []
    for entity in tqdm(add_pool.keys()):
        while add_pool[entity] > 0:
            # 对实体做oversampling
            one_x, one_y, one_length, one_inword = over_sampling(entity, data_train_path, n_tr_batches)
            all_x.append(one_x)
            all_y.append(one_y)
            all_length.append(one_length)
            all_inword.append(one_inword)

            add_pool[entity] -= 1
    return all_x, all_y, all_length, all_inword


def get_batch(data_path, data_id, is_submit=False):
    data = np.load(data_path + str(data_id) + '.npz')
    if not is_submit:
        return data['X'], data['inword'], data['len'], data['y']
    else:
        return data['X'], data['inword'], data['len'], data['belong'], data['comma']


def get_jieba_dict():
    _process = data_process.DataProcess()
    files_path = _process.y_files_path
    middle_path = DICT_PATH + '\dictionary.txt'
    # choose_entity = ['Test_Value', 'Level', 'Symptom']  # Test, raw data: <6%
    raw_entity_list = []
    count = 0
    target = 50
    for file_path in files_path:
        with open(file_path, 'rb') as f:
            file = f.read().decode('utf-8')
        file = re.split('\n', file)
        target_delete(file)
        for sentence in file:
            sentence_list = re.split('\t', sentence)
            raw_entity = sentence_list[-1]
            raw_entity_list.append(raw_entity)
            # entity_type = re.split('\s', sentence_list[1])[0]
            # print('entity type: %s, raw data: %s' % (entity_type, raw_entity))
    unique_with_count = Counter(raw_entity_list).most_common(28300)  # length=28272
    for data in unique_with_count:
        if data[1] > target and (data[0].count(' ') == 0):
            write = data[0] + ' ' + str(data[1]) + '\n'
            with codecs.open(middle_path, 'a', encoding='utf-8') as f:
                f.write(write)
            count += 1
    print('done, the file length is: ', count)


if __name__ == '__main__':
    # get_jieba_dict()
    pass
