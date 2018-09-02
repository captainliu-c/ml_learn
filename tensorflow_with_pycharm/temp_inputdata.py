import numpy as np


class MnistData(object):
    __data = np.loadtxt(r'C:\Users\Neo\ml_digit\data\train.csv', dtype="float32", delimiter=',', skiprows=1)
    __x = __data[:, 1:]
    __y = __data[:, 0]
    m, n = __x.shape

    x_tensor = __x
    y_tensor = np.zeros((m, 10), dtype="float64")
    count = 0
    for i in __y:
        y_tensor[count, int(i)] = 1.0
        count += 1

    def mini_batch(self, num):
        idx = 0
        while True:
            my_data = []
            if idx + num > self.m:
                idx = 0
            start = idx
            idx = idx + num
            print(idx)
            data_x = self.x_tensor[start:idx, :]
            data_y = self.y_tensor[start:idx, :]
            my_data.append(data_x)
            my_data.append(data_y)
            yield my_data


if __name__ == "__main__":
    mnist = MnistData()
    data = mnist.mini_batch(10)
    for i in range(5):
        y = next(data)
        # print(y)
        print('---------------')
