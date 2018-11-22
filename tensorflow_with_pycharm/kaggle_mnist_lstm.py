import tensorflow as tf
import temp_inputdata
from tensorflow.contrib import rnn

mydata = temp_inputdata.MnistData()

config = tf.ConfigProto(
    device_count={'CPU': 4},
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4)


def my_test():
    X = tf.constant(1.1, shape=[50, 28])  # batch=50 input_size=28
    outputs = list()

    def lstm():
        return rnn.BasicLSTMCell(36)  # hidden_size=36

    mult_lstm = rnn.MultiRNNCell([lstm() for _ in range(2)])  # 正确的生产参数的方式
    # print(lstm.state_size)
    state = mult_lstm.zero_state(50, dtype=tf.float32)
    for _ in range(2):
        output, state = mult_lstm(X, state)
        outputs.append(output)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())  # 对执行到此处的所有的变量进行initializer,下面的变量不进行
        for i in range(2):
            print(sess.run(outputs)[i][0, :])  # 没写reuse也OK，也许内部实现了


def my_test_reuse():
    with tf.variable_scope('fun'):
        v = tf.get_variable('v', [1])
        tf.get_variable_scope().reuse_variables()
        v1 = tf.get_variable('v', [1])
        assert v1 is v

    with tf.variable_scope('root'):
        assert tf.get_variable_scope().reuse == False
        with tf.variable_scope('foo'):
            assert tf.get_variable_scope().reuse == False
        with tf.variable_scope('foo', reuse=True):
            assert tf.get_variable_scope().reuse == True
            with tf.variable_scope('bar'):
                assert tf.get_variable_scope().reuse == True
        assert tf.get_variable_scope().reuse == False

    with tf.variable_scope('foo'):
        with tf.name_scope('bar'):
            v = tf.get_variable('v', [1])
            x = tf.add(v, 1)
        assert v.name == 'foo/v:0'
        print(x.name)  # foo/bar/Add:0
        # assert x.name == 'foo/bar/add'


# 创建超参 #
lr = 1e-3
hidden_size = 128
layer_nums = 2

classes = 10
time_steps = 28
input_size = 28

# 创建数据占位符 #
X = tf.placeholder(tf.float32, shape=[None, 784])  # shape = batch, 784
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # shape = batch, 10
input_X = tf.reshape(X, [-1, time_steps, input_size])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 一定要标注出batch_size的shape是[]


# 创建lstm模型，获得假设函数 | hidden size=128, 两层 #
def create_cell(hidden_size, keep_prob):
    cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    drop_cell = rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return drop_cell


mult_lstm = rnn.MultiRNNCell([create_cell(hidden_size, keep_prob) for _ in range(layer_nums)])
state = mult_lstm.zero_state(batch_size, dtype=tf.float32)
outputs, hs = tf.nn.dynamic_rnn(cell=mult_lstm, inputs=input_X, initial_state=state)
output = outputs[:, -1, :]  # 理论上来说，最后一个是一个batch,128的输出。需要一个layer把128转换成classes[10]

# 创建layer，将输出映射成10类 #
W = tf.Variable(tf.truncated_normal([hidden_size, classes], stddev=0.1))  # stddev=0.1
b = tf.Variable(tf.constant(0.1, shape=[classes]))  # 0.1
h_func = tf.matmul(output, W)+b
h_func = tf.nn.softmax(h_func)

# 创建cost func，使用梯度下架 #
cost_func = -tf.reduce_sum(y_*tf.log(h_func))
train_step = tf.train.AdamOptimizer(lr).minimize(cost_func)
correct_prediction = tf.equal(tf.argmax(h_func, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


def main():
    # 读取数据 #
    train_batch_size = 128
    train_data = mydata.mini_batch(train_batch_size)
    test_data = mydata.get_test()
    test_batch_size = test_data[0].shape[0]

    # create session #
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession(config=config)
    init.run()
    for i in range(5000):
        batch = next(train_data)
        train_step.run(feed_dict={
            X: batch[0], y_: batch[1], keep_prob: 0.5, batch_size: train_batch_size})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={X: batch[0], y_: batch[1], keep_prob: 1.0, batch_size: train_batch_size})
            print('the step is %d the correct prediction is %g:' % (i, train_accuracy))
    test_accuracy = accuracy.eval(
        feed_dict={X: test_data[0], y_: test_data[1], keep_prob: 1.0, batch_size: test_batch_size})
    print('the test accuracy is:', test_accuracy)
    sess.close()
    return None


if __name__ == '__main__':
    my_test()