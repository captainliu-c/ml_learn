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
        self.__input_path = r'C:\Users\Neo\Desktop\ruijin_round1_train1_20181010'
        self.__output_path = r'C:\Users\Neo\Desktop\process_data'
        self.__file_types = [ 'TXT', 'ann']
        self.__validation_percentage = 20
        self.__test_percentage = 0
        self.__commas = r'。'
        self.__sentence_mini_length = 10
        self.__control = 'off'

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
            print('the previous is %s, and the current value is %s' % (previous_value, value))
        else:
            print('the value must be on or off')

    def __get_files(self, file_type):
        assert file_type in self.__file_types
        file_glob = os.path.join(self.__input_path, '*.'+file_type)
        return glob.glob(file_glob)

    def __x_data_process(self, x_path):  # 注意下x和y的对应关系不要错位
        x_data = []
        index = 0
        with open(x_path, 'rb') as x_file:
            x_sentences = x_file.read().decode('utf-8')
            x_sentences = re.sub(r'\n', '', x_sentences)  # 原始文本乱换行，所以要把原本的换行干掉
            x_sentences = re.split(('%s' % self.commas), x_sentences)  # 以句号分割的句子组成的列表
            while '' in x_sentences:  # 去掉空行
                x_sentences.remove('')
            while index < len(x_sentences)-1:  # 检查由于self.__commas乱用，从而错误换行，最终出现异常短的句子
                if len(x_sentences[index]) < self.__sentence_mini_length:
                    print('the sentence[%s]is too short, the length is %d ' %
                          (x_sentences[index], len(x_sentences[index])))
                    x_data.append(str(x_sentences[index]+x_sentences[index+1]))
                    index += 1
                else:
                    x_data.append(x_sentences[index])
                index += 1
            x_data.append(x_sentences[-1])
            del x_sentences
        tools.check_sentence_length(x_data, self.__control)
        # 收集Test_Value所对应的实体包含的特殊符号, 得到set
        # 同时先按照ann文件制作y，然后同步删除x和y中同位置的相同符号[set除外]
        # 确定：1.标记方式 2.Test_Value中对应的特殊符号处理方法
        # 为什么要删除特殊符号，是因为每一个字符需要对应一个向量。特殊符号不需要？
        return x_data

    def __y_data_process(self, y):
        return y

    def get_data(self):
        for file_type in self.__file_types:
            print('The data begin to process which the type is %s' % file_type)
            if file_type == 'TXT':
                x_files = self.__get_files(file_type)
                x_datas = list(map(self.__x_data_process, x_files))
            else:
                y_files = self.__get_files(file_type)
                y_datas = list(map(self.__y_data_process, y_files))
            break
        # process_data = np.asarray([x_datas, y_datas])
        # np.save(self.__output_path, process_data)


def main():
    my_data_process = DataProcess()
    my_data_process.control = 'on'
    my_data_process.get_data()


if __name__ == '__main__':
    main()
