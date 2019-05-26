#pylint: skip-file
import os
import tensorflow as tf
from tensorflow.contrib import rnn
from load_data import load
from load_data import split33
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LOGDIR = "/tmp/final_project/33RLA_LLA_LC_RC_acc"
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
max_pool_size = 5
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
y_ = tf.placeholder(tf.float32, shape=[None, n_class])
x_reshape = tf.reshape(x, [-1, natt, nrow, 1])
print("x", x_reshape.get_shape().as_list())

#cnn
ndepth1 = 4
ndepth2 = 8
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



# attribute1
conv11 = conv_layer(xsplit1, kernel_size, max_pool_size, 1, ndepth1, "conv11")
conv12 = conv_layer(conv11, kernel_size, max_pool_size, ndepth1, ndepth2, "conv12")
print("conv12", conv12.get_shape().as_list())
# attribute2
conv21 = conv_layer(xsplit2, kernel_size, max_pool_size, 1, ndepth1, "conv21")
conv22 = conv_layer(conv21, kernel_size, max_pool_size, ndepth1, ndepth2, "conv22")
print("conv22", conv22.get_shape().as_list())
# attribute3
conv31 = conv_layer(xsplit3, kernel_size, max_pool_size, 1, ndepth1, "conv31")
conv32 = conv_layer(conv31, kernel_size, max_pool_size, ndepth1, ndepth2, "conv32")
print("conv32", conv32.get_shape().as_list())
# attribute4
conv41 = conv_layer(xsplit4, kernel_size, max_pool_size, 1, ndepth1, "conv41")
conv42 = conv_layer(conv41, kernel_size, max_pool_size, ndepth1, ndepth2, "conv42")
print("conv42", conv42.get_shape().as_list())
# attribute5
conv51 = conv_layer(xsplit5, kernel_size, max_pool_size, 1, ndepth1, "conv51")
conv52 = conv_layer(conv51, kernel_size, max_pool_size, ndepth1, ndepth2, "conv52")
print("conv52", conv52.get_shape().as_list())
# attribute6
conv61 = conv_layer(xsplit6, kernel_size, max_pool_size, 1, ndepth1, "conv61")
conv62 = conv_layer(conv61, kernel_size, max_pool_size, ndepth1, ndepth2, "conv62")
print("conv62", conv62.get_shape().as_list())
# attribute7
conv71 = conv_layer(xsplit7, kernel_size, max_pool_size, 1, ndepth1, "conv71")
conv72 = conv_layer(conv71, kernel_size, max_pool_size, ndepth1, ndepth2, "conv72")
print("conv72", conv72.get_shape().as_list())
# attribute8
conv81 = conv_layer(xsplit8, kernel_size, max_pool_size, 1, ndepth1, "conv81")
conv82 = conv_layer(conv81, kernel_size, max_pool_size, ndepth1, ndepth2, "conv82")
print("conv82", conv82.get_shape().as_list())
# attribute9
conv91 = conv_layer(xsplit9, kernel_size, max_pool_size, 1, ndepth1, "conv91")
conv92 = conv_layer(conv91, kernel_size, max_pool_size, ndepth1, ndepth2, "conv92")
print("conv92", conv92.get_shape().as_list())
# attribute10
conv101 = conv_layer(xsplit10, kernel_size, max_pool_size, 1, ndepth1, "conv101")
conv102 = conv_layer(conv101, kernel_size, max_pool_size, ndepth1, ndepth2, "conv102")
print("conv102", conv102.get_shape().as_list())
# attribute11
conv111 = conv_layer(xsplit11, kernel_size, max_pool_size, 1, ndepth1, "conv111")
conv112 = conv_layer(conv111, kernel_size, max_pool_size, ndepth1, ndepth2, "conv112")
print("conv112", conv112.get_shape().as_list())
# attribute12
conv121 = conv_layer(xsplit12, kernel_size, max_pool_size, 1, ndepth1, "conv121")
conv122 = conv_layer(conv121, kernel_size, max_pool_size, ndepth1, ndepth2, "conv122")
print("conv122", conv122.get_shape().as_list())
# # attribute13
# conv131 = conv_layer(xsplit13, kernel_size, max_pool_size, 1, ndepth1, "conv131")
# conv132 = conv_layer(conv131, kernel_size, max_pool_size, ndepth1, ndepth2, "conv132")
# print("conv132", conv132.get_shape().as_list())
# # attribute14
# conv141 = conv_layer(xsplit14, kernel_size, max_pool_size, 1, ndepth1, "conv141")
# conv142 = conv_layer(conv141, kernel_size, max_pool_size, ndepth1, ndepth2, "conv142")
# print("conv142", conv142.get_shape().as_list())
# # attribute15
# conv151 = conv_layer(xsplit15, kernel_size, max_pool_size, 1, ndepth1, "conv151")
# conv152 = conv_layer(conv151, kernel_size, max_pool_size, ndepth1, ndepth2, "conv152")
# print("conv122", conv122.get_shape().as_list())
# # attribute16
# conv161 = conv_layer(xsplit16, kernel_size, max_pool_size, 1, ndepth1, "conv161")
# conv162 = conv_layer(conv161, kernel_size, max_pool_size, ndepth1, ndepth2, "conv162")
# print("conv162", conv162.get_shape().as_list())
# # attribute17
# conv171 = conv_layer(xsplit17, kernel_size, max_pool_size, 1, ndepth1, "conv171")
# conv172 = conv_layer(conv171, kernel_size, max_pool_size, ndepth1, ndepth2, "conv172")
# print("conv172", conv172.get_shape().as_list())
# # attribute18
# conv181 = conv_layer(xsplit18, kernel_size, max_pool_size, 1, ndepth1, "conv181")
# conv182 = conv_layer(conv181, kernel_size, max_pool_size, ndepth1, ndepth2, "conv182")
# print("conv182", conv182.get_shape().as_list())
# # attribute19
# conv191 = conv_layer(xsplit19, kernel_size, max_pool_size, 1, ndepth1, "conv191")
# conv192 = conv_layer(conv191, kernel_size, max_pool_size, ndepth1, ndepth2, "conv192")
# print("conv192", conv192.get_shape().as_list())
# # attribute20
# conv201 = conv_layer(xsplit20, kernel_size, max_pool_size, 1, ndepth1, "conv201")
# conv202 = conv_layer(conv201, kernel_size, max_pool_size, ndepth1, ndepth2, "conv202")
# print("conv202", conv202.get_shape().as_list())
# # attribute21
# conv211 = conv_layer(xsplit21, kernel_size, max_pool_size, 1, ndepth1, "conv211")
# conv212 = conv_layer(conv211, kernel_size, max_pool_size, ndepth1, ndepth2, "conv212")
# print("conv212", conv212.get_shape().as_list())
# # attribute22
# conv221 = conv_layer(xsplit22, kernel_size, max_pool_size, 1, ndepth1, "conv221")
# conv222 = conv_layer(conv221, kernel_size, max_pool_size, ndepth1, ndepth2, "conv222")
# print("conv222", conv222.get_shape().as_list())
# # attribute23
# conv231 = conv_layer(xsplit23, kernel_size, max_pool_size, 1, ndepth1, "conv231")
# conv232 = conv_layer(conv231, kernel_size, max_pool_size, ndepth1, ndepth2, "conv232")
# print("conv232", conv232.get_shape().as_list())
# # attribute24
# conv241 = conv_layer(xsplit24, kernel_size, max_pool_size, 1, ndepth1, "conv241")
# conv242 = conv_layer(conv241, kernel_size, max_pool_size, ndepth1, ndepth2, "conv242")
# print("conv242", conv242.get_shape().as_list())


nfeature = natt * ndepth2

# rnn
concat = tf.concat([conv12, conv22, conv32, 
                    conv42, conv52, conv62, 
                    conv72, conv82, conv92, 
                    conv102, conv112, conv122,
                    # conv132, conv142, conv152,
                    # conv162, conv172, conv182,
                    # conv192, conv202, conv212,
                    # conv222, conv232, conv242
                    ], axis=3)

ntime = concat.get_shape().as_list()[2]
print("concat", concat.get_shape().as_list())
concat = tf.reshape(concat, [-1, 1 * ntime * nfeature])
print("concat", concat.get_shape().as_list())
inputs = tf.split(concat, num_or_size_splits=ntime, axis=1)
print("inputs", inputs[0].get_shape().as_list())
rnn_cell = rnn.BasicLSTMCell(rnn_size) #RNN cell
outputs, states = rnn.static_rnn(rnn_cell, inputs, dtype=tf.float32)


#top_layer
fc1size = 33
fc1 = fc_layer(outputs[-1], rnn_size, fc1size, "fc1")
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
        [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y_: batch[1]})
        test_accuracy = sess.run(accuracy, feed_dict={x: batch1[0], y_: batch1[1]})
        print('step %d, testing accuracy %g, traning accuracy %g' % (i, test_accuracy, train_accuracy))
        writer.add_summary(s, i)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
end = time.time()      
print('Done training! Total time %f s'%(end - start))
print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

sess.close()