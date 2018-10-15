import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

config = tf.ConfigProto(
    device_count={'CPU': 4},
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4,
)

INPUT_PATH = './data/flower_photos'
OUTPUT_PATH = './data/flower_processed_data.npy'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_lists(sess, test_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_PATH)]
    is_root_dir = True

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_PATH, dir_name, '*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:  # 如果file_list为空，则跳过本次循环
            continue

        for file_name in file_list:
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()  # 读出来是什么格式的文件
            image = tf.image.decode_jpeg(image_raw_data)  # 以jpeg的方式来读取吗？
            if image.dtype != tf.float32:  # 为什么有dtype
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

        chance = np.random.randint(100)
        if chance < validation_percentage:
            validation_images.append(image_value)
            validation_labels.append(current_label)
        elif chance < (validation_percentage + test_percentage):
            testing_images.append(image_value)
            testing_labels.append(current_label)
        else:
            training_images.append(image_value)
            training_labels.append(current_label)
    current_label += 1

    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])


def main():
    with tf.Session(config=config) as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        np.save(OUTPUT_PATH, processed_data)


if __name__ == '__main__':
    main()