import numpy as np
import os.path
import glob
import re
from bilstm_cnn_crf import tools
import pickle
from collections import Counter
from tqdm import tqdm


class DataProcess(object):
    def __init__(self):
        if __name__ == '__main__':
            self.__root_path = os.getcwd()
        else:  # module调用
            self.__root_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")))
        self.__input_path = self.__root_path + r'/data/raw_data/train/'
        self.__submit_input_path = self.__root_path + r'/data/raw_data/submit/'
        self.__train_output_path = self.__root_path + r'/data/process_data/train/'
        self.__valid_output_path = self.__root_path + r'/data/process_data/validation/'
        self.__submit_output_path = self.__root_path + r'/data/process_data/result/'
        self.__oversampling_path = self.__root_path + r'/data/process_data/oversampling/'
        self.__middle_path = self.root_path + r'/middle/'
        self.__file_types = ['TXT', 'ann']
        self.__validation_percentage = 0.20
        self.__test_percentage = 0
        self.__commas = ['。', ]  # ','
        self.__padding_comma = '#'
        self.__time_step = 150  # train set中超过这个数的比例是0.117835
        self.__dictionary_size = 3000  # real 3242, 经过根据句子长度截断处理后变为3147
        self.__batch_size = 200
        self.__submit_batch_size = 500
        self.__tags_prefixes = ['B_', 'I_']
        self.__tags_list = ['Disease', 'Reason', 'Symptom', 'Test', 'Test_Value', 'Drug', 'Frequency', 'Amount',
                            'Method', 'Treatment', 'Operation', 'Anatomy', 'Level', 'Duration', 'SideEff']
        self.__setting_control = {'to_wrap_word': True, 'tag_is_int': True, 'is_self_outdata': True}
        self.__check_control = {'sentence_length': False, 'entity_and_rawdata': False, 'final_data': False,
                                'check_2long_sentence': False, 'show_file_name': True}
        self.__tags = self.__create_tags(self.__tags_list)
        self.__y_files_path = self.__get_files(self.file_types[1], self.input_path)
        self.submit_files_path = self.__get_files(self.file_types[0], self.__submit_input_path)
        self.file2batch_relationship = dict(zip(list(map(lambda x: re.split('\\\\', x)[-1], self.submit_files_path)),
                                                range(len(self.submit_files_path))))
        self.__file2batch_relationship_reverse = dict(zip(self.file2batch_relationship.values(),
                                                          self.file2batch_relationship.keys()))

    @property
    def root_path(self):
        return self.__root_path

    @property
    def submit_input_path(self):
        return self.__submit_input_path

    @property
    def input_path(self):
        return self.__input_path

    @input_path.setter
    def input_path(self, path):
        if type(path) == 'str':
            self.__input_path = path
        else:
            raise ValueError('the path must be a string')

    @property
    def train_output_path(self):
        return self.__train_output_path

    @property
    def valid_output_path(self):
        return self.__valid_output_path

    @property
    def oversampling_path(self):
        return self.__oversampling_path

    @property
    def middle_path(self):
        return self.__middle_path

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
    def submit_batch_size(self):
        return self.__submit_batch_size

    @property
    def commas(self):
        return self.__commas

    @property
    def padding_comma(self):
        return self.__padding_comma

    @property
    def batch_size(self):
        return self.__batch_size

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

    @property
    def file2batch_relationship_reverse(self):
        return self.__file2batch_relationship_reverse

    @property
    def tags_list(self):
        return self.__tags_list

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
    def check_entity_and_raw_data(data, x_file, check_control=True):
        if check_control:
            begin_index, end_index = data[2], data[3]
            if len(data) > 5:  # 带换行的实体  ['T341', 'Test', 6560, 6561, 6562, 6563, '体重']
                raw_x_file = str(''.join(x_file[begin_index:end_index]) + ' ' + ''.join(x_file[data[4]:data[5]]))
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
        tags = dict(zip(keys, list(range(len(keys)))))
        return tags

    def __get_files(self, file_type, file_path):
        assert file_type in self.file_types
        file_glob = os.path.join(file_path, '*.'+file_type)
        return glob.glob(file_glob)

    def __entity2tags(self, x_file, data):
        key_begin, key_in = str(self.tags_prefixes[0] + data[1]), str(self.tags_prefixes[1] + data[1])
        begin_index, end_index = data[2], data[3]
        loop_count = int((len(data)-3)/2)-1

        self.check_entity_and_raw_data(data, x_file, self.check_control['entity_and_rawdata'])
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
        y_with_tag = x_file[:]
        # 对实体进行标记
        for y_data in sorted_y:
            y_with_tag = self.__entity2tags(y_with_tag, y_data)
        # 对剩余内容进行标记
        for data in y_with_tag:
            current_index = y_with_tag.index(data)
            if current_index not in entities_index:
                if data not in self.commas:
                    # if data != '\n':
                    y_with_tag[current_index] = self.tags['Other']
        return y_with_tag

    def __get_dictionary(self, x_sub):
        char_dictionary = {}
        pre_dictionary = []  # pre_dictionary = [('UNK', -1)]
        total_chars = tools.flatten(x_sub)
        # print('The real dictionary size=', len(Counter(total_chars)))
        most_common_chars = Counter(total_chars).most_common(self.dictionary_size)
        pre_dictionary.extend(most_common_chars)
        for char, _ in pre_dictionary:
            if char != self.padding_comma:  # '#'会映射为0
                char_dictionary[char] = len(char_dictionary)+1
        return char_dictionary

    def _make_batch(self, datas, output_path, batch_size, validation_percentage=0.0, batch_num=0):
        train_batch_num = batch_num
        valid_batch_num = 0
        sample_num = len(datas['X'])
        for start in tqdm(range(0, sample_num, batch_size)):
            train_batch_path = output_path + str(train_batch_num) + '.npz'
            valid_batch_path = self.valid_output_path + str(valid_batch_num) + '.npz'
            batch = {'X': -1, 'y': -1, 'len': -1, 'inword': -1, 'belong': -1, 'comma': -1}
            end = min(start+batch_size, sample_num)
            for key, value in datas.items():
                batch[key] = value[start:end]
            if np.random.rand()+1e-8 < validation_percentage:  # 1e-8保证在oversampling期间，不会有batch保存到valid中
                np.savez(valid_batch_path, X=batch['X'], y=batch['y'], len=batch['len'], inword=batch['inword'],
                         belong=batch['belong'], comma=batch['comma'])
                valid_batch_num += 1
            else:
                np.savez(train_batch_path, X=batch['X'], y=batch['y'], len=batch['len'], inword=batch['inword'],
                         belong=batch['belong'], comma=batch['comma'])
                train_batch_num += 1
        print('Finish! Train batch number is %d, validation batch number is %d' % (train_batch_num, valid_batch_num))

    def check_data_classes(self):
        count_by_classes = [0 for _ in range(len(self.__tags_list))]
        for y_file_path in self.y_files_path:
            with open(y_file_path, 'rb') as f:
                f = f.read().decode('utf-8')
            file = re.split('\n', f)
            _len = len(file)
            tools.target_delete(file)
            assert len(file) == _len-1
            for _data in file:
                data = re.split('\s', _data, maxsplit=2)[1]
                count_by_classes[self.__tags_list.index(data)] += 1

        total_count = sum(count_by_classes)
        print('total count is:', total_count)
        rate = [x/total_count for x in count_by_classes]
        tags_index = 0
        result = {}
        while tags_index < len(self.__tags_list):
            result[self.__tags_list[tags_index]] = count_by_classes[tags_index]
            print(self.__tags_list[tags_index], ': %.3f' % rate[tags_index], '| count: ', count_by_classes[tags_index])
            tags_index += 1
        return result

    def make_data(self):
        """
        1. 对原始txt转化成list
        2. 对ann文件进行处理，获得有序的实体的index
        2.1 实体的index有存在于两行的问题
        2.2 实体的index有在相同的index，存在两个实体的问题:不跳过
        3. 首先标记实体，接着标记other
        4. 删除换行符[未删除]，并根据句号进行拆分句子
        """
        middle_path = self.middle_path
        if not os.path.exists(middle_path+'x_sub.npy'):
            x_sub, y_sub = [], []
            sentence_lengths = []
            result_data = {'x': x_sub, 'y': y_sub}
            for y_file_path in tqdm(self.y_files_path):
                start_index = 0
                x_file_path = re.sub('%s' % self.file_types[1], '%s' % self.file_types[0], y_file_path)
                x_file_name = re.split('\\\\', x_file_path)[-1]
                if True in self.check_control.values():
                    print('-The file is[%s]' % x_file_name)
                with open(y_file_path, 'rb') as y_file:
                    y_datas = y_file.read().decode('utf-8')
                with open(x_file_path, 'rb') as x_file:
                    x_file = x_file.read().decode('utf-8')

                # 获得一些基本数据：sorted_y、entities_index
                sorted_y = self.__sort_y(y_datas)
                entities_index = self.__collect_entities_index(sorted_y)
                # 获得y_with_tag，
                y_with_tag = self.__add_tags(sorted_y, x_file, entities_index)  # target_delete(y_with_tag, target='\n')
                # 获得x data
                x_raw_data = [x for x in x_file]  # target_delete(x_raw_data, target='\n')
                assert len(y_with_tag) == len(x_raw_data)

                # 拆分句子，并对句子进行padding处理
                raw_data = {'x': x_raw_data, 'y': y_with_tag}
                index = 0
                while index < len(y_with_tag):  # 如果最后一个字符不是self.commas的话，最后一段data会少append进去
                    y_data = raw_data['y'][index]  # 但是似乎影响也不大
                    if y_data in self.commas:
                        assert y_data == raw_data['x'][index]
                        padding_count = len([x for x in range(self.time_step-(index-start_index))])
                        end_index = min(start_index+self.time_step, index)
                        for name in ['x', 'y']:
                            _sentence = raw_data[name][start_index:end_index]
                            for _ in range(padding_count):
                                if name == 'x':
                                    _sentence.append(self.padding_comma)
                                else:
                                    _sentence.append(0)  # x和y都padding成0
                            result_data[name].append(_sentence)
                        sentence_lengths.append(end_index-start_index)
                        start_index = index+1
                    index += 1
            tools.check_sentence_length(y_sub, control=self.check_control['sentence_length'])
            np.save(middle_path+'x_sub.npy', x_sub)
            np.save(middle_path+'y_sub.npy', y_sub)
            np.save(middle_path+'sentence_lengths.npy', sentence_lengths)
        else:
            x_sub = np.load(middle_path+'x_sub.npy')
            y_sub = np.load(middle_path+'y_sub.npy')
            sentence_lengths = np.load(middle_path+'sentence_lengths.npy')

        # 检查是否包含空元素，以及位置是否一一对应
        for i in y_sub:
            if len(i) == 0:
                empty_index = y_sub.index(i)
                assert len(x_sub[empty_index]) == 0  # 保证x_sub对应的空与y_sub的空是相同的位置
        x_sub = [x for x in x_sub if x != []]
        y_sub = [y for y in y_sub if y != []]
        assert len(x_sub) == len(y_sub)  # 如果x_sub对应的index集合是y_sub对应的超集,就会出问题
        tools.check_too_long_sentence(y_sub, self.time_step, self.check_control['check_2long_sentence'])
        # 将x映射为数字
        char_dictionary = self.__get_dictionary(x_sub)
        if not os.path.exists(middle_path+'vocab_dict.pickle'):
            with open(middle_path+'vocab_dict.pickle', 'wb') as handle:
                pickle.dump(char_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        x_sub2 = []
        for x_sentence in x_sub:
            temp_sentence = []
            for x_char in x_sentence:
                temp_sentence.append(char_dictionary.get(x_char, 0))  # padding 和 UNK都填成0
            x_sub2.append(temp_sentence)
        # 获得x的词位置标记
        x_with_inword_tag = tools.add_in_word_index(x_sub)
        # 将x_sub2和y_sub保存成按batch的npz
        datas = {'X': x_sub2, 'y': y_sub, 'len': sentence_lengths, 'inword': x_with_inword_tag}
        self._make_batch(datas, self.train_output_path,
                         batch_size=self.batch_size, validation_percentage=self.validation_percentage)
        return None

    def make_submit_data(self, char_dictionary):  # char_dictionary
        submit_final = []
        submit_sentence_length = []
        file_belong = []  # len=5064
        result = []
        is_comma = []

        for submit_file_path in tqdm(self.submit_files_path):
            print('begin to process the file: %s' % re.split('\\\\', submit_file_path)[-1])
            with open(submit_file_path, 'rb') as file:
                file = file.read().decode('utf-8')
            raw_data = [x for x in file]  # string to list
            # 拆分句子
            start_index, index = 0, 0
            while index < len(raw_data):
                data = raw_data[index]
                if data in self.commas:
                    # print('i find the commas in the index: %d' % index)
                    sentence_length = index-start_index
                    while sentence_length > 0:
                        end_index = min(start_index + self.time_step, index)
                        padding_count = len([x for x in range(self.time_step - sentence_length)])
                        _sentence = raw_data[start_index:end_index]
                        _sentence.extend([self.padding_comma for _ in range(padding_count)])

                        submit_final.append(_sentence)  # append data
                        submit_sentence_length.append(self.time_step - padding_count)  # append length
                        is_comma.append(int(padding_count > 0))  # append is_comma tag
                        file_belong.append(self.file2batch_relationship[re.split('\\\\', submit_file_path)[-1]])

                        sentence_length -= self.time_step
                        start_index += self.time_step - padding_count
                    start_index = index + 1
                index += 1
            # 确认是否有最后一段数据。看了下目前的数据有3个文档拥有最后一项，由于内容无意义，所以未处理
            if (start_index < index) and len(raw_data[start_index:]) != 1:
                print('see:', ''.join(raw_data[start_index:]))
            # tools.check_sentence_wrap_and_padding(submit_final, submit_sentence_length, is_comma)
        # 检查空元素
        for sentence in submit_final:
            if len(sentence) == 0:
                raise ValueError('there is empty sentence in data')
        print('total length is: ', len(submit_final))
        # 映射为数字
        for sentence in submit_final:
            temp_sentence = []
            for submit_char in sentence:
                temp_sentence.append(char_dictionary.get(submit_char, 0))  # padding 和 UNK都填成0
            result.append(temp_sentence)
        # 获得in word index tag
        submit_with_inword_tag = tools.add_in_word_index(submit_final)
        # make batch
        sub_datas = {'X': result, 'len': submit_sentence_length, 'belong': file_belong,
                     'comma': is_comma, 'inword': submit_with_inword_tag}
        self._make_batch(sub_datas, self.__submit_output_path, batch_size=self.submit_batch_size)

    def make_oversampling_batch(self):
        target = 2000  # 按entity数量小于2000来处理
        check_result = self.check_data_classes()
        over_x, over_y, over_length, over_inword = tools.make_oversampling(check_result, self.train_output_path,
                                                                           self.tags, self.tags_prefixes[0], target)
        n_batch_start = len(os.listdir(self.train_output_path))
        over_datas = {'X': over_x, 'y': over_y, 'len': over_length, 'inword': over_inword}
        self._make_batch(over_datas, self.oversampling_path, batch_size=self.batch_size, batch_num=n_batch_start)


def main():
    data_process = DataProcess()

    # 生成train和validation数据
    # data_process.make_data()
    # # 生成submit数据
    with open(data_process.middle_path + 'vocab_dict.pickle', 'rb') as handle:
        vocab_dict = pickle.load(handle)
    data_process.make_submit_data(vocab_dict)
    # 根据实际情况决定进行oversampling
    # data_process.make_oversampling_batch()


if __name__ == '__main__':
    main()
