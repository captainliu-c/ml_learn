import tensorflow as tf
import temp_inputdata
import mnist_inference
import os

config = tf.ConfigProto(
    device_count={'CPU': 4},
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4,
)

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULAR_DECAY = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './path/to/model/'
MODEL_NAME = 'model.ckpt'


def train(train_data):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], 'X')
    y_ = tf.placeholder(tf.int32, [None, mnist_inference.OUTPUT_NODE], 'Y')
    global_steps = tf.Variable(0, trainable=False)  # 如何自增的？
    y = mnist_inference.inference(x, wb=REGULAR_DECAY)  # 顺序的问题??!

    """准备好variable average op"""
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_steps)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())  # 滑动平均后的weight是如何被训练过程中使用的呢？

    """准备好train step op"""
    cross = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    lost = tf.reduce_mean(cross) + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_steps, 100, LEARNING_RATE_DECAY)  # 100?
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(lost, global_step=global_steps)

    """关联variable average op、train step op"""
    with tf.control_dependencies([variable_averages_op, train_step]):
        train_op = tf.no_op(name='train')

    """create session"""
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        train_data = train_data.mini_batch(BATCH_SIZE)
        for i in range(TRAINING_STEPS):
            batch = next(train_data)
            _, lost_value, step = sess.run([train_op, lost, global_steps], feed_dict={x: batch[0], y_: batch[1]})
            if i % 1000 == 0:
                print('the step is %d , the lost is %g' % (step, lost_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)


def main():
    data = temp_inputdata.MnistData()
    train(data)


if __name__ == '__main__':
    main()
