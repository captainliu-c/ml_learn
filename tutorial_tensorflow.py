import tensorflow as tf
import numpy as np


# 基本的Op的概念
a = tf.add(3, 5)
with tf.Session() as sess:
    print(sess.run(a))


# 稍微复杂一些的OP和常量
input1 = tf.constant(3.0)
input2 = tf.constant(4.0)
input3 = tf.constant(5.0)
intermed = tf.add(input1, input2)
mul = tf.multiply(intermed, input3)
with tf.Session() as sess:
    result = sess.run([intermed, mul])
    print(result)


# 变量的概念
x = tf.Variable(3, name = 'x')
y = x*5
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(y)


# 变量的应用，计数
count = tf.Variable(0, name = 'counter')  
one = tf.constant(1)
new_value = tf.add(count, one)
update = tf.assign(count, new_value)  
init_op = tf.global_variables_initializer()  
with tf.Session() as sess:
    sess.run(init_op)
    print(count)
    for i in range(3):
        sess.run(update)
        print(sess.run(count))


# 一会回归的例子
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300
b = tf.Variable(tf.zeros(1))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(0, 201):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run([W,b]))