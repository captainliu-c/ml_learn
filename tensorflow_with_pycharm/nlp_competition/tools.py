import numpy as np


def check_sentence_length(sentences, control='off'):
    if control == 'on':
        print('The check process is running. We will check the length of each of the sentence')
        length_sentences = list(map(len, sentences))
        index_and_length = dict(zip(length_sentences, range(len(length_sentences))))
        length_sentences.sort()
        print('the 10 minimum length is %s ,and the index of the sentence is %s' %
              (str(length_sentences[:10]), str(list(map(index_and_length.get, length_sentences[:10])))))
        print('the 10 maximum length is %s ,and the index of the sentence is %s' %
              (str(length_sentences[-10:]), str(list(map(index_and_length.get, length_sentences[-10:])))))
    else:
        return None
