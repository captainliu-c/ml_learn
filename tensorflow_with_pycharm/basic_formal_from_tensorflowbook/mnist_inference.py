import tensorflow as tf

INPUT_NODE = 784
HIDDEN_LAYER_NODE = 500
OUTPUT_NODE = 10


def _variable_on_cpu(name, shape, initializer):  # 不需要dtype吗
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wb):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))

    if wb:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wb)
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(inputs, wb=0.0):
    with tf.variable_scope('hidden_layer'):
        weights = _variable_with_weight_decay('weights', [INPUT_NODE, HIDDEN_LAYER_NODE], 0.1, wb)
        biases = _variable_on_cpu('biases', [HIDDEN_LAYER_NODE], tf.constant_initializer(0.1))
        hidden_layer = tf.nn.relu((tf.matmul(inputs, weights) + biases), 'hidden_layer')

    with tf.variable_scope('output'):
        weights = _variable_with_weight_decay('weights', [HIDDEN_LAYER_NODE, OUTPUT_NODE], 0.1, wb)
        biases = _variable_on_cpu('biases', [OUTPUT_NODE], tf.constant_initializer(0.1))
        output = tf.matmul(hidden_layer, weights) + biases  # 未加relu

    return output
