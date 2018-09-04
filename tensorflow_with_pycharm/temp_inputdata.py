import numpy as np
from sklearn.model_selection import train_test_split


class MnistData(object):
    def __init__(self):  # 注意split
        self.__train_data = np.loadtxt(r'E:\kaggle_data\mnist\train.csv', dtype="float32", delimiter=',', skiprows=1)
        self.__test_data = np.loadtxt(r'E:\kaggle_data\mnist\test.csv', dtype="float32", delimiter=',', skiprows=1)
        self.__sub_data = np.loadtxt(r'E:\kaggle_data\mnist\sample_submission.csv',
                                     dtype="int", delimiter=',', skiprows=1)
        self.__x = self.__train_data[:, 1:]
        self.__y = self.__train_data[:, 0]
        self.__size = 0.2
        self.__types = len(set(self.__y.flatten()))

    @property
    def train_x(self):
        return self.__x

    @property
    def train_y(self):
        return self.__y

    @property
    def types(self):
        return self.__types

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, value):
        if value < 0.0 or value > 1.0:
            raise ValueError('value must be in [0,1]')
        if not isinstance(value, float):
            raise ValueError('value must be float')
        self.__size = value

    # 对x做归一化处理, return x
    @staticmethod
    def __regulate(x):
        regular_x = np.multiply(x, 1.0/255.0)
        return regular_x

    # 按照8:2来拆分数据， return x_train, x_test, y_train, y_test
    def __split(self):
        x, y = self.__regulate(self.__x), self.__y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.__size, random_state=42)
        return x_train, x_test, y_train, y_test

    # 将y向量构造成y矩阵, test data可被外部访问 , return x_train_t, x_test_t, y_train_t, y_test_t
    def __make_tensor(self, tot="train"):
        x_train, x_test, y_train, y_test = self.__split()
        x_train_t = x_train
        x_test_t = x_test
        y_train_t = np.zeros((y_train.shape[0], self.__types), dtype="float64")
        y_test_t = np.zeros((y_test.shape[0], self.__types), dtype="float64")
        count_train, count_test = 0, 0
        for i in y_train:
            y_train_t[count_train, int(i)] = 1.0
            count_train += 1
        for i in y_test:
            y_test_t[count_test, int(i)] = 1.0
            count_test += 1
        if tot == "train":
            return x_train_t, y_train_t
        elif tot == "test":
            return x_test_t, y_test_t

    # use x_train_t, y_train_t, 每次大循环开始没有随机x和y, 官方提供的方法好像是对x和y进行切片
    def mini_batch(self, num):
        x_batch, y_batch = self.__make_tensor()
        idx = 0
        while True:
            my_data = []
            if idx + num > x_batch.shape[0]:
                idx = 0
            start = idx
            idx = idx + num
            data_x = x_batch[start:idx, :]
            data_y = y_batch[start:idx, :]
            my_data.append(data_x)
            my_data.append(data_y)
            yield my_data

    def get_test(self):
        my_test = []
        x_test, y_test = self.__make_tensor(tot="test")
        my_test.append(x_test)
        my_test.append(y_test)
        return my_test

    def sub_data(self):
        test_x = self.__regulate(self.__test_data)
        return test_x

    # 将y矩阵转换回向量，同时将结果以csv的格式保存到本地

    def transform_y(self, y):
        result = np.argmax(y, axis=1)
        # csv = np.c_(self.__sub_data[:, 1], result)
        return result


if __name__ == "__main__":
    mnist = MnistData()
    # print('types:', mnist.types)
    # mnist.size=0.3
    # print('size:', mnist.size)
    data = mnist.mini_batch(20)
    batch = next(data)
