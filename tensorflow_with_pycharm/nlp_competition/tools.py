import numpy as np
import jieba
from collections import Iterable


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


def main():
    pass


if __name__ == '__main__':
    print(list(jieba.cut('我是大白菜')))
