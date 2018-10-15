import tensorflow
import numpy as np
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
    def sentence_mini_length(self):
        return self.__sentence_mini_length

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

    def __data_process(self, x_path):  # 注意下x和y的对应关系不要错位
        print('\n-We will process the file:%s' % x_path)
        data = []
        index = 0
        with open(x_path, 'rb') as x_file:
            x_sentences = x_file.read().decode('utf-8')
            x_sentences = re.sub(r'\n', '', x_sentences)  # 原始文本乱换行，所以要把原本的换行干掉
            x_sentences = re.split(('%s' % self.commas), x_sentences)  # 以句号分割的句子组成的列表
            tools.blank_delete(x_sentences)
            while index < len(x_sentences)-1:  # 检查由于self.__commas乱用，从而错误换行，最终出现异常短的句子
                if len(x_sentences[index]) < self.__sentence_mini_length:
                    print('--The sentence[%s]is too short, the length is %d ' %
                          (x_sentences[index], len(x_sentences[index])))
                    data.append(str(x_sentences[index]+x_sentences[index+1]))
                    index += 1
                else:
                    data.append(x_sentences[index])
                index += 1
            data.append(x_sentences[-1])
            del x_sentences
        tools.check_sentence_length(data, self.__control)
        # 收集Test_Value所对应的实体包含的特殊符号, 得到set
        # 同时先按照ann文件制作y，然后同步删除x和y中同位置的相同符号[set除外]
        # 确定：1.标记方式 2.Test_Value中对应的特殊符号处理方法
        # 为什么要删除特殊符号，是因为每一个字符需要对应一个向量。特殊符号不需要？
        return data  # 以句子为元素组成的list

    def __entity2tag(self, y_data):  # 由于超过10个了，如果仍然要切片的话，需要用特殊的分隔符。或者直接以列表的形式保存。
        word_wrap = False
        for item in y_data:
            if ';' in item:
                word_wrap = True
        if word_wrap:
            # print('look: ', y_data)
            i_index = int(y_data[4])  # ['T34', 'Symptom', '353', '356;357', '358', '年龄较', '大']
        else:
            i_index = int(y_data[3])
        b_index = int(y_data[2])
        entity_type = y_data[1]
        key_begin = str(self.tags_prefixes[0]+entity_type)
        key_in = str(self.tags_prefixes[1]+entity_type)
        return str(self.tags[key_begin] + (self.tags[key_in]*(i_index-b_index-1)))

    def __add_tags(self, y_paths):
        y_datas_list = []
        with open(y_paths, 'rb') as y_file:
            y_datas = y_file.read().decode('utf-8')
        with open(y_paths.replace(self.file_types[1], self.file_types[0]), 'rb') as x_file:
            x_sentences = x_file.read().decode('utf-8')
        y_datas = re.split(r'\n', y_datas)
        tools.blank_delete(y_datas)
        for item in y_datas:  # y_datas_list[0] = ['T1', 'Disease', '1845', '1850', '1型糖尿病']
            y_datas_list.append(re.split(r'\s+', item))

        # 根据item[1]的类型对item[2]~[3]位置的文字进行标注
        # print(x_sentences[353:356])
        previous_length = len(x_sentences)
        for y_data in y_datas_list:
            entity2tag = self.__entity2tag(y_data)
            print('The entity is%s and the entity code is %s' % (str(y_data), entity2tag))
            # 直接修改x_sentences，用切片
        assert previous_length == len(x_sentences)

    def __x_data_process(self, x):
        return x

    def __y_data_process(self, y):
        return y

    def get_data(self):
        # for file_type in self.__file_types:
        #     print('The data begin to process which the type is %s' % file_type)
        #     if file_type == 'TXT':
        #         x_files = self.__get_files(file_type)
        #         datas = list(map(self.__data_process, x_files))  # 对原始数据做预处理
        #         x_datas = list(map(self.__x_data_process, datas))
        #     else:
        #         y_files = self.__get_files(file_type)
        #         y_datas = list(map(self.__y_data_process, y_files))
        #     break  # test
        # process_data = np.asarray([x_datas, y_datas])
        # np.save(self.__output_path, process_data)

        x_files = self.__get_files(self.__file_types[0])
        y_files = self.__get_files(self.__file_types[1])
        self.__add_tags(y_files[0])


def main():
    my_data_process = DataProcess()
    # my_data_process.control = 'on'
    # my_data_process.get_data()
    for item in my_data_process.tags.items():
        print(item)


if __name__ == '__main__':
    main()
