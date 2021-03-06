import tensorflow as tf
import temp_inputdata

mydata = temp_inputdata.MnistData()
ckpt_path = './ckpt/test-model.ckpt'
config = tf.ConfigProto(
    device_count={'CPU': 4},
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4,
    log_device_placement=True
)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 卷积层1
W_conv1 = weight_variable([5, 5, 1, 32])  # from 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 卷积层2
W_conv2 = weight_variable([5, 5, 32, 64])  # from 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# flatten层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# flatten
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 全连接
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 获得数据
data = mydata.mini_batch(50)
test_data = mydata.get_test()
sub = mydata.sub_data()
saver = tf.train.Saver()  # 位置要在创建变量的下面

# 创建session，开始跑
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()
for i in range(10000):
    batch = next(data)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.4})
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
save_path = saver.save(sess, ckpt_path, global_step=1)
print("the path of the model is:", save_path)
sess.close()

# 需要一个保存model的功能，直接在内部跑，过于占用内存 
with tf.Session(config=config) as sess2:
    saver.restore(sess2, ckpt_path + '-' + str(1))
    print("test accuracy %g" % accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1], keep_prob: 1.0}))

# 本地机子内存小，只有8个G，只能test data和sub数据都比较大，放在一个session里面，内存容易超 | 没啥用，该崩还是崩
with tf.Session(config=config) as sess3:
    saver.restore(sess3, ckpt_path + '-' + str(1))
    result = y_conv.eval(feed_dict={x: sub, keep_prob: 1.0})
    mydata.transform_y(result)
