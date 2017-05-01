import tensorflow as tf
import numpy as np

import random, math

from tensorflow.examples.tutorials.mnist import input_data

from utilities import *
import sys

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32) # dropout 


BATCH_SIZE = 500

init = tf.contrib.layers.xavier_initializer 
cinit = tf.constant_initializer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

s = tf.placeholder(tf.float32) # beginning: between 0 and 1
c = tf.placeholder(tf.float32) # capacity (end): between 0 and 1

def model(input):
    x_image = tf.reshape(input, [-1, 28, 28, 1])

    with tf.variable_scope("conv1"):
        fw = tf.get_variable("weights", [5, 5, 1, 32], initializer=init())
        w = fw

        fb = tf.get_variable("bias", [32], initializer=cinit(0.1))
        b = fb

        hconv1 = tf.nn.relu(conv2d(x_image, w) + b)
        hpool1 = max_pool_2x2(hconv1)

    with tf.variable_scope("conv2"):
        fw = tf.get_variable("weights", [5, 5, 32, 64], initializer=init())
        w = fw

        fb = tf.get_variable("bias", [64], initializer=cinit(0.1))
        b = fb

        hconv2 = tf.nn.relu(conv2d(hpool1, w) + b)
        hpool2 = max_pool_2x2(hconv2)

    with tf.variable_scope("fc"):
        fw = tf.get_variable("weights", [7 * 7 * 64, 1024], initializer=init())
        w = fw

        fb = tf.get_variable("bias", [1024], initializer=cinit(0.1))
        b = fb

        hpool2flat = tf.reshape(hpool2, [-1, 7 * 7 * 64])

        hfc1 = tf.nn.relu(tf.matmul(hpool2flat, w) + b)
        hfc1drop = tf.nn.dropout(hfc1, keep_prob)

    with tf.variable_scope("output"):
        fw = tf.get_variable("weights", [1024, 10], initializer=init())
        w = fw

        fb = tf.get_variable("bias", [10], initializer=cinit(0.1))
        b = fb

        yconv = tf.matmul(hfc1drop, w) + b
    
    return yconv

y_conv = model(x)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in xrange(1000):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if i%100 == 0:
            sys.stdout.flush()
            formatstr = "%d, %f, %f"
            test_acc = accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            train_acc = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})

            print formatstr % (i, test_acc, train_acc)
            #print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
