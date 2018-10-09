import tensorflow as tf
import mnist_inference
import mnist_train
import temp_inputdata
import time
from tensorflow.python import pywrap_tensorflow

config = tf.ConfigProto(
    device_count={'CPU': 4},
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4,
)
EVA_INTERVAL_SEC = 10


def check_checkpoint():
    ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        # print(reader.get_tensor(key))


def evaluate(eval_data):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], 'X')
        y_ = tf.placeholder(tf.int32, [None, mnist_inference.OUTPUT_NODE], 'Y')

        """评估模型：计算准确率"""
        y = mnist_inference.inference(x)
        correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        """关联滑动平均, variables_to_restore"""
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        saver = tf.train.Saver(variable_averages.variables_to_restore())  # 索引到weight和bias

        while True:
            with tf.Session() as sess:  # 每次循环都创建和销毁一个session吗
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: eval_data[0], y_: eval_data[1]})
                    print('after %s steps, the accuracy is %g' % (global_step, accuracy_score))
                else:
                    print('no checkpoint file found')
                    return
            time.sleep(EVA_INTERVAL_SEC)


def main():
    eval_data = temp_inputdata.MnistData().get_test()
    evaluate(eval_data)


if __name__ == '__main__':
    # check_checkpoint()
    main()
