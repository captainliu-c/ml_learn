import numpy as np
from collections import Iterable


def check_sentence_length(sentences, control='off'):
    if control == 'on':
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


def target_delete(the_list, target=''):
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

