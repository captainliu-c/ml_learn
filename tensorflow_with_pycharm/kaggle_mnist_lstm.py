import tensorflow as tf
from tensorflow.contrib import rnn

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


# 测一下reuse对普通方式创建的变量的效果
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


# 创建超参
lr = 1e-3
hidden_size = 128
batch_size = 50
layer_nums = 2
classes = 10
time_steps = 28
input_size = 28

# 创建数据占位符
X = tf.placeholder(tf.float32, shape=[])  # shape = batch, 784
y_ = tf.placeholder(tf.float32, shape=[])  # shape = batch, 10
input_X = tf.reshape(X, [-1, time_steps, input_size])
keep_prob = tf.placeholder(tf.float32)


# 创建lstm模型，获得假设函数 | hidden size=128, 两层
def create_cell(hidden_size, keep_prob):
    cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    drop_cell = rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return drop_cell


mult_lstm = rnn.MultiRNNCell([create_cell(hidden_size, keep_prob) for _ in range(layer_nums)])
state = mult_lstm.zero_state(batch_size, dtype=tf.float32)
outputs, hs = tf.nn.dynamic_rnn(cell=mult_lstm, inputs=input_X, initial_state=state)
output = outputs[:, -1, :]  # 理论上来说，最后一个是一个batch,128的输出。需要一个layer把128转换成classes[10]

# 创建layer，将输出映射成10类


# 创建cost func，使用梯度下架


# create session

if __name__ == '__main__':
    pass

