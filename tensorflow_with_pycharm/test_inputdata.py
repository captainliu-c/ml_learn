import tensorflow as tf
import numpy as np
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch = mnist.train.next_batch(50)
x = batch[0]
y = batch[1]
print(x.shape)
for i in range(10):
    print(x[:, i])

if __name__ == "__main__":
    print("?")


