# import tensorflow
# import numpy as np
import os.path
import glob
import re
import tools

"""
按照文件名，分别处理data和lable
处理的方式暂时pass。以ndrray保存文件。
"""


class DataProcess(object):
    def __init__(self):
        self.__input_path = r'C:\Users\nhn\Desktop\ruijin_round1_train1_20181010'
        self.__output_path = r'C:\Users\nhn\Desktop\process_data'
        self.__file_types = ['TXT', 'ann']
        self.__validation_percentage = 20
        self.__test_percentage = 0
        self.__commas = r'。'
        self.__sentence_mini_length = 10
        self.__control = 'off'
        self.__tags_list = ['Disease', 'Reason', 'Symptom', 'Test', 'Test_Value', 'Drug', 'Frequency', 'Amount',
                            'Method', 'Treatment', 'Operation', 'Anatomy', 'Level', 'Duration', 'SideEff']
        self.__tags_prefixes = ['B_', 'I_']
        self.__tags = self.__create_tags(self.__tags_list)
        self.__y_files_path = self.__get_files(self.file_types[1])
        # self.__special_commas = '！。？，、；：“”（）《》〈〉【】〔〕…—,.一%\[-\]'  # 记得转义]

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

    # @property
    # def sentence_mini_length(self):
    #     return self.__sentence_mini_length

    @property
    def control(self):
        return self.__control

    @control.setter
    def control(self, value):
        if value in ['on', 'off']:
            previous_value = self.__control
            self.__control = value
            print('The previous is %s, and the current value is %s' % (previous_value, value))
        else:
            print('The value must be on or off')

    @property
    def tags(self):
        return self.__tags

    @property
    def tags_prefixes(self):
        return self.__tags_prefixes

    @property
    def y_files_path(self):
        return self.__y_files_path

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
    def __sort_y(y_datas):
        y = []
        y_datas = re.split('\n', y_datas)
        tools.target_delete(y_datas)  # 好像不好使
        for item in y_datas:  # 记得把x的/n删除 | 这种情况下换行符不算实体
            if ';' in item:
                temp = re.split('\s+', item)
                fix_y = temp
                y_ = fix_y[:3]
                y_.extend(re.split(';', fix_y[3]))
                y_.append(fix_y[4])
                y_.append(''.join(fix_y[5:]))
                for i in range(4):
                    y_[i+2] = int(y_[i+2])
            else:
                y_ = re.split('\s+', item)  # T110	Test_Value 2402 2413	<3.3 mmol/L | 这样的数据会多分出来一部分
                if len(y_) > 5:             # 此处需要把空格加回去，也就是此处的空格算实体
                    temp = y_
                    y_ = y_[:4]
                    y_.append(str(temp[4]+' '+temp[5]))
                    for i in range(2):
                        y_[i+2] = int(y_[i+2])
                else:
                    for i in range(2):
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
    def check_y(y_with_tag, entities_index):
        for y in y_with_tag:
            if not y.isdigit():
                if not y.isspace():
                    print('-There is a wrong char[%s], the index is %d' % (y, y_with_tag.index(y)))
                    print('--The data is in entities index:', y in entities_index)

    def __create_tags(self, tags_list):  # BIO标注集, 总计共30类，超过10的怎么标注
        keys = ['Other']
        for tag in tags_list:
            for prefix in self.tags_prefixes:
                keys.append(str(prefix+tag))
        tags = dict(zip(keys, list(map(str, range(len(keys))))))
        return tags

    def __get_files(self, file_type):
        assert file_type in self.file_types
        file_glob = os.path.join(self.input_path, '*.'+file_type)
        return glob.glob(file_glob)

    def __entity2tags(self, x_file, data):
        key_begin = str(self.tags_prefixes[0] + data[1])
        key_in = str(self.tags_prefixes[1] + data[1])
        begin_index = data[2]
        end_index = data[3]

        if len(data) > 5:  # 带换行的实体  ['T341', 'Test', 6560, 6561, 6562, 6563, '体重']
            begin_index_2 = data[4]
            end_index_2 = data[5]
            x_file[begin_index] = self.tags[key_begin]
            for i in range(begin_index+1, end_index):
                x_file[i] = self.tags[key_in]
            for j in range(begin_index_2, end_index_2):
                x_file[j] = self.tags[key_in]
        else:
            # 不带换行的实体 ['T346', 'Test', 6621, 6626, 'HBA1C'], 直接标注
            if data[4] != ''.join(x_file[begin_index:end_index]):
                print('the y data is: ', data)
                print('--there is different from y | x: ', data[4], ' | ', ''.join(x_file[begin_index:end_index]))
            assert data[4] == ''.join(x_file[begin_index:end_index])

            x_file[begin_index] = self.tags[key_begin]
            for j in range(end_index - begin_index - 1):
                x_file[begin_index + 1 + j] = self.tags[key_in]
        return x_file

    def __add_tags(self, sorted_y, x_file, entities_index):  # 判断是否是相同实体重复标注、判断是否是换行实体、对实体进行标注
        x_file = self.__file2char(x_file)
        # print('-1-The length of the x file is %d' % len(x_file))
        index = 1
        count_skip = 0
        y_with_tag = self.__entity2tags(x_file, sorted_y[0])
        while index < len(sorted_y):
            pre_data = sorted_y[index-1]
            data = sorted_y[index]
            if data[2] == pre_data[2]:
                count_skip += 1
            else:
                y_with_tag = self.__entity2tags(y_with_tag, data)
            index += 1
        print('--We have skip %d datas, because of the same index have two entities' % count_skip)
        for data in y_with_tag:
            current_index = y_with_tag.index(data)
            if current_index not in entities_index:
                if data != self.commas:
                    if not data.isspace():
                        y_with_tag[current_index] = self.tags['Other']
        # print('-2-The length of the x file is %d' % len(y_with_tag))
        return y_with_tag

    def get_data(self):
        """
        1. 对原始txt转化成list
        2. 对ann文件进行处理，获得有序的实体的index
        2.1 实体的index有存在于两行的问题
        2.2 实体的index有在相同的index，存在两个实体的问题
        3. 首先标记实体，接着标记other
        4. 删除空格和换行符，并根据句号进行拆分句子
        """
        y_sub = []
        for y_file_path in self.y_files_path:
            y_final = []
            start_index = 0
            x_file_path = re.sub('%s' % self.file_types[1], '%s' % self.file_types[0], y_file_path)
            with open(y_file_path, 'rb') as y_file:
                y_datas = y_file.read().decode('utf-8')
                sorted_y = self.__sort_y(y_datas)
                entities_index = self.__collect_entities_index(sorted_y)
            with open(x_file_path, 'rb') as x_file:
                x_file = x_file.read().decode('utf-8')
                # print('-0-The length of the x file is %d' % len(x_file))
                print('-The file is[%s]' % re.split('\\\\', x_file_path)[-1])
                y_with_tag = self.__add_tags(sorted_y, x_file, entities_index)
            tools.target_delete(y_with_tag, target=' ')
            tools.target_delete(y_with_tag, target='\n')
            index = 0
            while index < len(y_with_tag):
                data = y_with_tag[index]
                if data == self.commas:
                    y_final.append(y_with_tag[start_index:index])
                    start_index = index+1
                index += 1
            tools.target_delete(y_final)
            tools.check_sentence_length(y_final, control=self.control)
            # 对不符合长度的sentence进行处理，丢弃或部位
            y_sub.append(y_final)


def main():
    my_data_process = DataProcess()
    my_data_process.control = 'on'
    my_data_process.get_data()


if __name__ == '__main__':
    main()
