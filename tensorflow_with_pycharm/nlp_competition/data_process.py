# import tensorflow as tf
import numpy as np
import os.path
import glob
import re
import tools
from collections import Counter
from tqdm import tqdm

"""
按照文件名，分别处理data和lable
处理的方式暂时pass。以ndrray保存文件。
"""


class DataProcess(object):
    def __init__(self):
        self.__input_path = r'C:\Users\nhn\Desktop\ruijin_round1_train1_20181010'
        self.__output_path = r'C:\Users\nhn\Desktop\process_data\\'
        self.__file_types = ['TXT', 'ann']
        self.__validation_percentage = 20
        self.__test_percentage = 0
        self.__commas = ['。', ]  # ','
        # self.__sentence_mini_length = 10
        self.__time_step = 150  # train set中超过这个数的比例是0.117835
        self.__dictionary_size = 3000  # real 3242
        self.__tags_prefixes = ['B_', 'I_']
        self.__tags_list = ['Disease', 'Reason', 'Symptom', 'Test', 'Test_Value', 'Drug', 'Frequency', 'Amount',
                            'Method', 'Treatment', 'Operation', 'Anatomy', 'Level', 'Duration', 'SideEff']
        self.__setting_control = {'to_wrap_word': True, 'tag_is_int': False, 'is_self_outdata': True}
        self.__check_control = {'sentence_length': False, 'entity_and_rawdata': False, 'final_data': False}
        self.__tags = self.__create_tags(self.__tags_list)
        self.__y_files_path = self.__get_files(self.file_types[1])

    @property
    def input_path(self):
        return self.__input_path

    @input_path.setter
    def input_path(self, path):
        if type(path) == 'str':  # 可以加一些其他的判断
            self.__input_path = path
        else:
            raise ValueError('the path must be a string')

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        if type(path) == 'str':  # 可以加一些其他的判断
            self.__output_path = path
        else:
            raise ValueError('the path must be a string')

    @property
    def file_types(self):
        return self.__file_types

    @property
    def validation_percentage(self):
        return self.__validation_percentage

    @validation_percentage.setter
    def validation_percentage(self, value):
        if (value > 100) or (value < 0):
            raise ValueError('the value must be in (0, 100)')
        else:
            self.__validation_percentage = value

    @property
    def test_percentage(self):
        return self.__test_percentage

    @test_percentage.setter
    def test_percentage(self, value):
        if (value > 100) or (value < 0):
            raise ValueError('the value must be in (0, 100)')
        else:
            self.__test_percentage = value

    @property
    def commas(self):
        return self.__commas

    @property
    def tags(self):
        return self.__tags

    @property
    def tags_prefixes(self):
        return self.__tags_prefixes

    @property
    def y_files_path(self):
        return self.__y_files_path

    @property
    def dictionary_size(self):
        return self.__dictionary_size

    @property
    def check_control(self):
        return self.__check_control

    @property
    def setting_control(self):
        return self.__setting_control

    @property
    def time_step(self):
        return self.__time_step

    @staticmethod
    def __file2char(file):
        result = []
        for char in file:
            result.append(char)
        return result

    @staticmethod
    def __open_files(files):
        files_list = []
        for file in files:
            with open(file, 'rb') as f:
                files_list.append(f.read().decode('utf-8'))
        return files_list

    @staticmethod
    def check_y(y_with_tag, entities_index):
        for y in y_with_tag:
            if not y.isdigit():
                if not y.isspace():
                    print('-There is a wrong char[%s], the index is %d' % (y, y_with_tag.index(y)))
                    print('--The data is in entities index:', y in entities_index)

    @staticmethod
    def __sort_y(y_datas):
        y = []
        y_datas = re.split('\n', y_datas)
        tools.target_delete(y_datas)
        for item in y_datas:
            wrap_count = len(re.findall('\d+;\d+', item))
            y_ = re.split('\s+', item, maxsplit=(4+wrap_count))
            temp = y_[:]
            # 前3个 +中间N个 +后两个
            y_ = y_[:3]
            for j in range(wrap_count):
                y_.extend(re.split(';', temp[3+j]))  # ['T34', 'Symptom', '353', '356;357', '358', '年龄较', '大']
            y_.extend(temp[-2:])
            # str2int
            for i in range(2+(wrap_count*2)):
                y_[i+2] = int(y_[i+2])
            y.append(y_)
        y.sort(key=lambda x: x[2])
        return y

    @staticmethod
    def __collect_entities_index(sorted_y):
        entities_index = []
        for data in sorted_y:
            begin_index, end_index = data[2], data[3]
            if len(data) > 5:  # ['T341', 'Test', 6560, 6561, 6562, 6563, '体重']
                begin_index_2, end_index_2 = data[4], data[5]
                entities_index.extend([x for x in range(begin_index, end_index)])
                entities_index.extend([x for x in range(begin_index_2, end_index_2)])
            else:
                entities_index.extend([x for x in range(begin_index, end_index)])
        # entities_index = set(entities_index)
        return entities_index

    @staticmethod
    def check_entity_and_raw_dada(data, x_file, check_control=True):
        if check_control:
            begin_index, end_index = data[2], data[3]
            if len(data) > 5:  # 带换行的实体  ['T341', 'Test', 6560, 6561, 6562, 6563, '体重']
                raw_x_file = str(''.join(x_file[begin_index:end_index]) + ' ' + ''.join(x_file[data[4]:data[5]]))
                # raw_x_file只考虑了一个;出现的情况，所以在检查多个的时候会出问题
                # 123_8.TXT的T219和41.TXT的T223存在这个情况，但是标注的时候是按照多个;来标注的
                temp = re.findall('[A-Z]_[A-Z][a-z]+', raw_x_file)
                if len(temp) != data[5] - begin_index:
                    if data[-1] != raw_x_file:
                        print('--the y data is: ', data)
                        print('--[wrap]there is different from y | x: ', data[-1], ' | ', raw_x_file)
            else:  # 不带换行的实体 ['T346', 'Test', 6621, 6626, 'HBA1C'], 直接标注
                raw_x_file = ''.join(x_file[begin_index:end_index])
                temp = re.findall('[A-Z]_[A-Z][a-z]+', raw_x_file)
                if len(temp) != end_index - begin_index:
                    if data[-1] != raw_x_file:
                        print('--the y data is: ', data)
                        print('--there is different from y | x: ', data[-1], ' | ',
                              ''.join(x_file[begin_index:end_index]))

    def __create_tags(self, tags_list):  # BIO标注集, 总计共30类
        keys = ['Other']
        for tag in tags_list:
            for prefix in self.tags_prefixes:
                keys.append(str(prefix+tag))
        if self.setting_control['tag_is_int']:
            tags = dict(zip(keys, list(map(str, range(len(keys))))))
        else:
            tags = dict(zip(keys, keys))
        return tags

    def __get_files(self, file_type):
        assert file_type in self.file_types
        file_glob = os.path.join(self.input_path, '*.'+file_type)
        return glob.glob(file_glob)

    def __entity2tags(self, x_file, data):
        key_begin, key_in = str(self.tags_prefixes[0] + data[1]), str(self.tags_prefixes[1] + data[1])
        begin_index, end_index = data[2], data[3]
        loop_count = int((len(data)-3)/2)-1

        self.check_entity_and_raw_dada(data, x_file, self.check_control['entity_and_rawdata'])
        x_file[begin_index] = self.tags[key_begin]  # 带换行的实体  ['T341', 'Test', 6560, 6561, 6562, 6563, '体重']
        for i in range(begin_index + 1, end_index):
            x_file[i] = self.tags[key_in]
        for _ in range(loop_count):
            begin_index_2, end_index_2 = 4, 5
            for j in range(data[begin_index_2], data[end_index_2]):
                x_file[j] = self.tags[key_in]
            begin_index_2 += 2
            end_index_2 += 2
        return x_file

    def __add_tags(self, sorted_y, x_file, entities_index):
        """判断是否是相同实体重复标注、判断是否是换行实体、对实体进行标注"""
        x_file = self.__file2char(x_file)
        index = 1
        count_skip = 0
        # 对实体进行标记
        y_with_tag = self.__entity2tags(x_file, sorted_y[0])
        while index < len(sorted_y):
            pre_data = sorted_y[index-1]
            data = sorted_y[index]
            if data[2] == pre_data[2]:  # 确认是否是同一个index对应的两个实体
                count_skip += 1
            else:
                y_with_tag = self.__entity2tags(y_with_tag, data)
            index += 1
        # print('--We have skip %d datas, because of the same index have two entities' % count_skip)
        # 对剩余内容进行标记
        for data in y_with_tag:
            current_index = y_with_tag.index(data)
            if current_index not in entities_index:
                if data not in self.commas:
                    if not data.isspace():
                        y_with_tag[current_index] = self.tags['Other']
        return y_with_tag

    def make_data(self):
        """
        1. 对原始txt转化成list
        2. 对ann文件进行处理，获得有序的实体的index
        2.1 实体的index有存在于两行的问题
        2.2 实体的index有在相同的index，存在两个实体的问题
        3. 首先标记实体，接着标记other
        4. 删除空格和换行符，并根据句号进行拆分句子
        """
        x_sub, y_sub = [], []
        for y_file_path in tqdm(self.y_files_path):
            y_final = []
            x_final = []
            start_index = 0
            x_file_path = re.sub('%s' % self.file_types[1], '%s' % self.file_types[0], y_file_path)
            with open(y_file_path, 'rb') as y_file:
                y_datas = y_file.read().decode('utf-8')
                sorted_y = self.__sort_y(y_datas)
                entities_index = self.__collect_entities_index(sorted_y)
            with open(x_file_path, 'rb') as x_file:
                x_file = x_file.read().decode('utf-8')
                x_file_name = re.split('\\\\', x_file_path)[-1]
                if True in self.check_control.values():
                    print('-The file is[%s]' % x_file_name)
                y_with_tag = self.__add_tags(sorted_y, x_file, entities_index)
            tools.target_delete(y_with_tag, target='\n')

            x_raw_data = [x for x in x_file]
            tools.target_delete(x_raw_data, target='\n')

            if self.setting_control['to_wrap_word']:
                index = 0
                while index < len(y_with_tag):
                    y_data = y_with_tag[index]
                    x_data = x_raw_data[index]
                    if y_data in self.commas:
                        y_final.append(y_with_tag[start_index:index])
                        assert x_data == y_data
                        x_final.append(x_raw_data[start_index:index])
                        start_index = index+1
                    index += 1
            else:
                y_final, x_final = y_with_tag, x_raw_data

            tools.check_sentence_length(y_final, control=self.check_control['sentence_length'])
            # 对不符合长度的sentence进行处理，丢弃或部位 | 未进行

            # path = os.path.join(self.output_path, x_file_name)
            # if self.setting_control['is_self_outdata']:
            #     # x_path = os.path.join(self.output_path, str('x_'+x_file_name[:-3]+'npy'))  #存早了
            #     # y_path = os.path.join(self.output_path, str('y_'+x_file_name[:-3]+'npy'))
            #     # np.save(x_path, x_data)
            #     # np.save(y_path, y_final)
            #     pass
            # else:
            #     with open(path, 'w', encoding='utf-8') as f:
            #         j = 0
            #         while j < len(x_data):
            #             if x_final[j] == self.commas[0]:  # 因为只是句号换行，所以不是in
            #                 f.write(x_final[j] + ' O' + '\n' + '\n')
            #             elif y_final[j] == self.tags['Other']:
            #                 f.write(x_final[j] + ' O' + '\n')
            #             else:
            #                 f.write(x_final[j]+' '+y_final[j]+'\n')
            #             j += 1
            x_sub.extend(x_final)
            y_sub.extend(y_final)
            assert len(x_sub) == len(y_sub)

        if type(y_sub[0]) == list:
            print('Attention: the y_final may have [[]] or [[[]]] data')
        else:
            tools.target_delete(y_sub)
            tools.target_delete(x_sub)

        sentences_count = len(y_sub)
        too_long_sentence = [i for i in list(map(len, y_sub)) if i > self.time_step]
        print('the amount of the data[sentence] is %d' % sentences_count)
        print('the rate of the too long sentence is %g, the number is %d'
              % (len(too_long_sentence)/sentences_count, len(too_long_sentence)))
        print(too_long_sentence)
        # char_dictionary = self.__get_dictionary(x_sub)
        # for item in char_dictionary.items():
        #     print(item)
        return None

    def __get_dictionary(self, x_sub):
        char_dictionary = {}
        pre_dictionary = [('UNK', -1)]
        total_chars = tools.flatten(x_sub)
        # print('The real dictionary size=', len(Counter(total_chars)))
        most_common_chars = Counter(total_chars).most_common(self.dictionary_size-1)
        pre_dictionary.extend(most_common_chars)
        print(pre_dictionary)
        for char, _ in pre_dictionary:
            char_dictionary[char] = len(char_dictionary)
        return char_dictionary


def main():
    my_data_process = DataProcess()
    my_data_process.make_data()


if __name__ == '__main__':
    main()
