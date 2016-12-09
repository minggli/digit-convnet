import tensorflow as tf
import pandas as pd
import numpy as np
from utilities import extract, batch_iter
import warnings
import os
import re


# params

dir_path = 'input/'
model_path = 'models/'
test = pd.read_csv('input/test.csv')
test.index += 1
train, label, data = extract('input/train.csv')
input_shape = (np.int32(test.shape[1] ** 0.5), np.int32(test.shape[1] ** 0.5))
m = test.shape[1]  # num of flat array
n = 10

# load image into tensor

sess = tf.Session()

# declare placeholders

x = tf.placeholder(dtype=tf.float32, shape=[None, m], name='feature')  # pixels as features
y_ = tf.placeholder(dtype=tf.float32, shape=[None, n], name='label')  # 99 classes in 1D tensor

# declare variables

W = tf.Variable(tf.zeros([m, n]))
b = tf.Variable(tf.zeros([n]))

y = tf.matmul(x, W) + b


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


# First Convolution Layer
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, input_shape[0], input_shape[1], 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, n])
b_fc2 = bias_variable([n])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

# load

saver = tf.train.Saver()
model_names = [i.name for i in os.scandir(model_path) if i.is_file() and i.name.endswith('.meta')]
loop_num = re.findall("[0-9]", model_names.pop())[0]
new_saver = tf.train.import_meta_graph(model_path + "model_loop_{0}.ckpt.meta".format(loop_num))
new_saver.restore(save_path=tf.train.latest_checkpoint(model_path), sess=sess)


ans = sess.run(tf.nn.softmax(y_conv), feed_dict={x: test, keep_prob: 1})

data = pd.DataFrame(data=ans, columns=label.columns, dtype=np.float32, index=test.index)
data.index.name = 'ImageId'
data['Label'] = data.idxmax(axis=1)
out = data['Label']
data.to_csv('prob.csv', encoding='utf-8', header=True, index=True)
out.to_csv('submission.csv', encoding='utf-8', header=True, index=True)

