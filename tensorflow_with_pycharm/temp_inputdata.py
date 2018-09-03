import numpy as np
from sklearn.model_selection import train_test_split


class MnistData(object):
    __data = np.loadtxt(r'C:\Users\Neo\ml_digit\data\train.csv', dtype="float32", delimiter=',', skiprows=1)
    __x = __data[:, 1:]
    __y = __data[:, 0]

    """
    官方处理的数据是形如这样的数据：
    0.         0.5176471  0.6745098  0.         0.         0.
    0.         0.         0.         0.         0.         0.
    0.         0.         0.         0.         0.         0.
    0.         0.         0.         0.         0.         0.32156864
    0.63529414 0.7960785  0.7960785  0.5568628  0.2392157  0.20000002
    0.98823535 0.32156864 0.         0.         0.         0.
    0.         0.         0.         0.         0.         0.
    0.         0.         0.         0.         0.         0.
    0.         0.         0.08235294 0.9960785  0.9921569  0.9960785
    按照常规的缩放方式，一定会把0给缩放了；但是看官方的数据格式，似乎只处理非0的值。
    关于这一规则的进一步探索似乎只有看源码
    方法中可以按照两个方式同时来做，一个是常规的把0也缩放，另外一个是按照官方提供的方式来做，看一下效果对比
    官方代码是按照全部乘以1/255来进行的
    """

    # 对x做归一化处理, return x, y
    def __regulate(self):
        x = np.multiply(self.__x, 1.0/255.0)
        y = self.__y
        return x, y

    # 按照8:2来拆分数据， return x_train, x_test, y_train, y_test
    def __split(self):
        size = 0.2
        x, y = self.__regulate()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)
        return x_train, x_test, y_train, y_test

    # 将y向量构造成y矩阵, test data可被外部访问 , return x_train_t, x_test_t, y_train_t, y_test_t
    def __make_tensor(self, tot="train"):
        x_train, x_test, y_train, y_test = self.__split()
        x_train_t = x_train
        x_test_t = x_test
        y_train_t = np.zeros((y_train.shape[0], 10), dtype="float64")
        y_test_t = np.zeros((y_test.shape[0], 10), dtype="float64")
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

    # use x_train_t, y_train_t, 每次大循环开始没有随机x和y
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


if __name__ == "__main__":
    mnist = MnistData()
    data = mnist.mini_batch(10)
    batch = next(data)
    print(batch[0])
