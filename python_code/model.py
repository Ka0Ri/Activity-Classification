#pylint: skip-file
import os
import tensorflow as tf
from tensorflow.contrib import rnn
from load_data import load
from load_data import split10
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LOGDIR = "/tmp/final_project/10LCacc"
import shutil
shutil.rmtree(LOGDIR)
# load data
path = "D:/final project/data/"
attribute = ["LCaccX", "LCaccY", "LCaccZ", "label"]

print("loading data")
train_data, test_data = load(path, attribute)
print("loading done! training size: %r, testing size: %r"%(np.shape(train_data), np.shape(test_data)))

kernel_size = 7
max_pool_size = 5
batch_size = 50
n_class = 10
natt = len(attribute) - 1
nrow = 200
rnn_size = 100
learning_rate = 0.01
moment_rate = 0.01


def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([1, kernel_size, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        # summary
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 1, max_pool_size, 1], strides=[1, 1, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(
            [size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        # summary
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)
        return act

tf.reset_default_graph()
sess = tf.Session()

#model
x = tf.placeholder(tf.float32, shape=[None, natt, nrow])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_reshape = tf.reshape(x, [-1, natt, nrow, 1])
print("x", x_reshape.get_shape().as_list())

#cnn
ndepth1 = 4
ndepth2 = 8
xsplit1, xsplit2, xsplit3 = tf.split(x_reshape, num_or_size_splits=natt, axis=1)
print("x_split", xsplit1.get_shape().as_list())
# attribute1
conv11 = conv_layer(xsplit1, 1, ndepth1, "conv11")
conv12 = conv_layer(conv11, ndepth1, ndepth2, "conv12")
print("conv12", conv12.get_shape().as_list())
# attribute2
conv21 = conv_layer(xsplit2, 1, ndepth1, "conv21")
conv22 = conv_layer(conv21, ndepth1, ndepth2, "conv22")
print("conv22", conv22.get_shape().as_list())
# attribute3
conv31 = conv_layer(xsplit3, 1, ndepth1, "conv31")
conv32 = conv_layer(conv31, ndepth1, ndepth2, "conv32")
print("conv32", conv32.get_shape().as_list())

nfeature = natt * ndepth2
# rnn
concat = tf.concat([conv12, conv22, conv32], axis=3)
ntime = concat.get_shape().as_list()[2]
print("concat", concat.get_shape().as_list())
concat = tf.reshape(concat, [-1, 1 * ntime * nfeature])
print("concat", concat.get_shape().as_list())
inputs = tf.split(concat, num_or_size_splits=ntime, axis=1)
print("inputs", inputs[0].get_shape().as_list())
rnn_cell = rnn.BasicLSTMCell(rnn_size) #LSTM cell
outputs, states = rnn.static_rnn(rnn_cell, inputs, dtype=tf.float32)

#top_layer
fc1 = fc_layer(outputs[-1], rnn_size, n_class, "fc1")

logits = fc1

#cost function
with tf.name_scope("xent"):
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_), name="xent")
    tf.summary.scalar("xent", xent)

with tf.name_scope("train"):
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(xent)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# create summary
summ = tf.summary.merge_all()

# create writer
writer = tf.summary.FileWriter(LOGDIR)

sess.run(tf.global_variables_initializer())

writer.add_graph(sess.graph)
batch1 = split10(test_data)
start = time.time()
print("training")
for i in range(1001):
    np.random.shuffle(train_data)
    train = train_data[0:batch_size, :,:]
    
    batch = split10(train)
    # batch1 = split(train_data[50:batch_size + 50, :,:])
    # write summary
    if i % 100 == 0:
        [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y_: batch[1]})
        test_accuracy = sess.run(accuracy, feed_dict={x: batch1[0], y_: batch1[1]})
        print('step %d, testing accuracy %g, traning accuracy %g' % (i, test_accuracy, train_accuracy))
        writer.add_summary(s, i)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
end = time.time()      
print('Done training! Total time %f s'%(end - start))
print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

sess.close()