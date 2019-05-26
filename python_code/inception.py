#pylint: skip-file
import os
import tensorflow as tf
from tensorflow.contrib import rnn
from load_data import load
from load_data import split33
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LOGDIR = "/tmp/final_project/inception_33RLA_LLA_LC_RC_acc"
import shutil
if(os.path.exists(LOGDIR)):
    shutil.rmtree(LOGDIR)
# load data
path = "D:/final project/all_data/"
attribute = ["RLAaccX", "RLAaccY", "RLAaccZ", 
            "LLAaccX", "LLAaccY", "LLAaccZ", 
            "LCaccX", "LCaccY", "LCaccZ", 
            "RCaccX", "RCaccY", "RCaccZ",
            # "RUAaccX", "RUAaccY", "RUAaccZ", 
            # "LUAaccX", "LUAaccY", "LUAaccZ", 
            # "LTaccX", "LTaccY", "LTaccZ", 
            # "RTaccX", "RTaccY", "RTaccZ", 
            "label"]

print("loading data")
train_data, test_data = load(path, attribute)
print("loading done! training size: %r, testing size: %r"%(np.shape(train_data), np.shape(test_data)))

kernel_size = 7
max_pool_size = 7
batch_size = 50
n_class = 33
natt = len(attribute) - 1
nrow = 200
rnn_size = 100
learning_rate = 0.01


def conv_layer(input, kernel_size, max_pool_size, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(
            [1, kernel_size, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        # summary
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 1, max_pool_size, 1], strides=[1, 1, 2, 1], padding="SAME")

def inception_module(input,size_in, size_out, name="inception"):
    with tf.name_scope(name):
        incepconv1 = conv_layer(input, 7, 7, size_in, size_out, name + "conv1")
        incepconv2 = conv_layer(input, 5, 5, size_in, size_out, name + "conv2")
        incepconv3 = conv_layer(input, 3, 3, size_in, size_out, name + "conv3")
        concat = tf.concat([incepconv1, incepconv2, incepconv3], axis=3)
        # return concat
        # max_pool = tf.nn.max_pool(concat, ksize=[1, 1, max_pool_size, 1], strides=[1, 1, 2, 1], padding="SAME")
        # w = tf.Variable(tf.truncated_normal([1, 1, size_out*3, size_out*3], stddev=0.1), name="W")
        # conv = tf.nn.conv2d(max_pool, w, strides=[1, 1, 1, 1], padding="SAME")
        # return conv
        return tf.nn.max_pool(concat, ksize=[1, 1, max_pool_size, 1], strides=[1, 1, 2, 1], padding="SAME")

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
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, natt, nrow])
y_ = tf.placeholder(tf.float32, shape=[None, n_class])
x_reshape = tf.reshape(x, [-1, natt, nrow, 1])
print("x", x_reshape.get_shape().as_list())

#cnn
ndepth1 = 4
# ndepth2 = 8
[
xsplit1, xsplit2, xsplit3, 
xsplit4, xsplit5, xsplit6, 
xsplit7, xsplit8, xsplit9, 
xsplit10, xsplit11, xsplit12
# xsplit13, xsplit14, xsplit15, 
# xsplit16, xsplit17, xsplit18, 
# xsplit19, xsplit20, xsplit21, 
# xsplit22, xsplit23, xsplit24
] = tf.split(x_reshape, num_or_size_splits=natt, axis=1)
print("x_split", xsplit1.get_shape().as_list())

################
# attribute1
incep1 = inception_module(xsplit1, 1 , ndepth1, "incep1")
print("incep1", incep1.get_shape().as_list())
# attribute2
incep2 = inception_module(xsplit2, 1 , ndepth1, "incep2")
print("incep2", incep2.get_shape().as_list())
# attribute3
incep3 = inception_module(xsplit3, 1 , ndepth1, "incep3")
print("incep3", incep3.get_shape().as_list())
# attribute4
incep4 = inception_module(xsplit4, 1 , ndepth1, "incep4")
print("incep4", incep4.get_shape().as_list())
# attribute5
incep5 = inception_module(xsplit5, 1 , ndepth1, "incep5")
print("incep5", incep5.get_shape().as_list())
# attribute6
incep6 = inception_module(xsplit6, 1 , ndepth1, "incep6")
print("incep6", incep6.get_shape().as_list())
# attribute7
incep7 = inception_module(xsplit7, 1 , ndepth1, "incep7")
print("incep7", incep7.get_shape().as_list())
# attribute8
incep8 = inception_module(xsplit8, 1 , ndepth1, "incep8")
print("incep8", incep1.get_shape().as_list())
# attribute9
incep9 = inception_module(xsplit9, 1 , ndepth1, "incep9")
print("incep9", incep9.get_shape().as_list())
# attribute10
incep10 = inception_module(xsplit10, 1 , ndepth1, "incep10")
print("incep10", incep10.get_shape().as_list())
# attribute11
incep11 = inception_module(xsplit11, 1 , ndepth1, "incep11")
print("incep11", incep11.get_shape().as_list())
# attribute12
incep12 = inception_module(xsplit12, 1 , ndepth1, "incep12")
print("incep12", incep12.get_shape().as_list())
################

nfeature = natt * ndepth1 * 3
# rnn

concat = tf.concat([incep1, incep2, incep3, 
                    incep4, incep5, incep6, 
                    incep7, incep8, incep9, 
                    incep10, incep11, incep12,
                    ], axis=3)

ntime = concat.get_shape().as_list()[2]
print("concat", concat.get_shape().as_list())
concat = tf.reshape(concat, [-1, 1 * ntime * nfeature])
print("concat", concat.get_shape().as_list())
inputs = tf.split(concat, num_or_size_splits=ntime, axis=1)
print("inputs", inputs[0].get_shape().as_list())
rnn_cell = rnn.BasicLSTMCell(rnn_size) #RNN cell
outputs, states = rnn.static_rnn(rnn_cell, inputs, dtype=tf.float32)

fc1_drop = tf.nn.dropout(outputs[-1], keep_prob)
#top_layer
fc1size = 33
fc1 = fc_layer(fc1_drop, rnn_size, fc1size, "fc1")
print("fc1", fc1.get_shape().as_list())
# relu1 = tf.nn.relu(fc1)
# fc2 = fc_layer(fc1, fc1size, n_class, "fc2")
# print("fc2", fc2.get_shape().as_list())
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
batch1 = split33(test_data)
start = time.time()
print("training")
for i in range(1501):
    np.random.shuffle(train_data)
    train = train_data[0:batch_size, :,:]
    
    batch = split33(train)
    # batch1 = split(train_data[50:batch_size + 50, :,:])
    # write summary
    if i % 100 == 0:
        [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        test_accuracy = sess.run(accuracy, feed_dict={x: batch1[0], y_: batch1[1], keep_prob: 1.0})
        print('step %d, testing accuracy %g, traning accuracy %g' % (i, test_accuracy, train_accuracy))
        writer.add_summary(s, i)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
end = time.time()      
print('Done training! Total time %f s'%(end - start))
print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

sess.close()