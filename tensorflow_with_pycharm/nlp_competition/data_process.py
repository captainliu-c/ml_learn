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
        self.__x_files_path = self.__get_files(self.file_types[0])
        self.__y_files_path = self.__get_files(self.file_types[1])
        self.__x_files = self.__open_files(self.x_files_path)  #
        # self.__x_total_index = list(map(tools.list_save_index, self.__open_files(self.x_files_path)))
        self.__tags_index = None  # range [[538,558], [143,145]...]
        self.__separator_comma_index = None  # point [2, 12, 23...]
        # self.__special_commas = '！！？＂＃＄％%＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》〔〕〚〛〜〝〞〟–—‘’‛“”„‟…‧﹏.、'
        self.__special_commas = '！。？！，、；：“”（）《》〈〉【】『』「」﹃﹄〔〕…—～﹏￥¨⋯)(,.一%\[-\];'  # 记得转义]
        self.__skip_comma = '#'

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

    @property
    def x_files_path(self):
        return self.__x_files_path

    @property
    def y_files_path(self):
        return self.__y_files_path

    @property
    def x_files(self):
        return self.__x_files

    @property
    def special_commas(self):
        return self.__special_commas

    @property
    def skip_comma(self):
        return self.__skip_comma

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

    def __data_process(self, x_path):  # 注意下x和y的对应关系不要错位 | 对数据做预处理
        print('\n-We will process the file:%s' % x_path)
        data = []
        index = 0
        with open(x_path, 'rb') as x_file:
            x_sentences = x_file.read().decode('utf-8')
            x_sentences = re.sub(r'\n', '', x_sentences)  # 原始文本乱换行，所以要把原本的换行干掉
            x_sentences = re.split(('%s' % self.commas), x_sentences)  # 以句号分割的句子组成的列表
            tools.target_delete(x_sentences)
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

    def __entity2tag(self, y_data):
        result = []
        word_wrap = False
        for item in y_data:
            if ';' in item:
                word_wrap = True
        if word_wrap:
            # print('look: ', y_data)
            i_index = int(y_data[4])-1  # ['T34', 'Symptom', '353', '356;357', '358', '年龄较', '大'] # 删除换行符的占位
        else:
            i_index = int(y_data[3])
        b_index = int(y_data[2])
        entity_type = y_data[1]
        key_begin = str(self.tags_prefixes[0]+entity_type)
        key_in = str(self.tags_prefixes[1]+entity_type)
        result.append(self.skip_comma)
        result.append(self.tags[key_begin])
        for i in range(i_index-b_index-1):
            result.append(self.skip_comma)
            result.append(self.tags[key_in])
        result.append(self.skip_comma)
        return ''.join(result), b_index, i_index

    def __add_tags(self, y_paths):
        y_datas_list = []
        with open(y_paths, 'rb') as y_file:
            y_datas = y_file.read().decode('utf-8')
        with open(y_paths.replace(self.file_types[1], self.file_types[0]), 'rb') as x_file:
            x_sentences = x_file.read().decode('utf-8')
        y_datas = re.split(r'\n', y_datas)
        tools.target_delete(y_datas)
        for item in y_datas:  # y_datas_list[0] = ['T1', 'Disease', '1845', '1850', '1型糖尿病']
            y_datas_list.append(re.split(r'\s+', item))

        # 根据item[1]的类型对item[2]~[3]位置的文字进行标注
        previous_length = len(x_sentences)
        for y_data in y_datas_list:
            entity2tag, b_index, i_index = self.__entity2tag(y_data)
            # print('The entity is%s and the entity code is %s' % (str(y_data), entity2tag))
            # 直接修改x_sentences，用切片
            x_sentences = x_sentences[:b_index]+entity2tag+x_sentences[i_index:]
            # 这样不行，由于每次追加entity2tag后，由于#的加入和忽略[;]导致文本整体的对应关系的变化，后续index无法对应正确文字
        assert previous_length == len(x_sentences)

    def __x_data_process(self, x):
        return x

    def __y_data_process(self, y):
        return y

    @staticmethod
    def __special_commas_index(x_file, clean_x):
        index = []
        for data in clean_x:
            index.append(x_file.index(data))  # 需要保证data是唯一的，要不然返回的是第一个匹配的
        return index

    def get_data(self):
        # 使用x_file添加tag， 注意i_index-1
        # 按照句号将list转换为shape = 句子数量，一行句子[一行句子合并成str]
        # 使用re.sub对空格和特殊符号进行删除[保留实体对应的符号]，shape仍为=句子数量，一行句子 | 如何保证tag不与文本中原有的数字混淆
        # 制作y，将句子映射为字符，并添加other tag
        # 制作x
        y_sub = []
        for y_file_path in self.y_files_path:
            # 先制作y
            # 根据x和y，对x进行标记
            y_with_tag = self.__add_tags(y_file_path)
            break


def main():
    my_data_process = DataProcess()
    # my_data_process.control = 'on'
    my_data_process.get_data()


if __name__ == '__main__':
    main()
