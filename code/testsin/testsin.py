import tensorflow as tf
import numpy as np

import random
import math

from utilities import *


# generate a batch of size num of example math.sin data into a numpy array
def gensin(num):
    inarr = np.float32(np.random.uniform(-2*math.pi, 2*math.pi, (1, num))).T
    outarr = np.sin(inarr)

    return inarr, outarr

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

e = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

FULL_CAPACITY = 1000

def model(input):
    init = tf.contrib.layers.xavier_initializer 
    with tf.variable_scope("network") as scope:
        fw1 = tf.get_variable("hweights", [1,FULL_CAPACITY], initializer=init())
        # only trains a subsection from b to e of the weights
        w1 = tf.slice(fw1, [0,b], [1,e])
        tw1 = tf.stop_gradient(tf.slice(fw1, [0,0], [1,b]))
        w1 = tf.concat([tw1, w1], 1)

        fb1 = tf.get_variable("hbias", [FULL_CAPACITY])
        tb1 = tf.stop_gradient(tf.slice(fb1, [0], [b]))
        b1 = tf.slice(fb1, [b], [e])
        b1 = tf.concat([tb1, b1], 0)

        h = tf.nn.relu(tf.matmul(input, w1) + b1)

        fw2 = tf.get_variable("oweights", [FULL_CAPACITY,1], initializer=init())
        w2 = tf.slice(fw2, [b,0], [e,1])
        tw2 = tf.stop_gradient(tf.slice(fw2, [0,0], [b,1]))
        w2 = tf.concat([tw2, w2], 0)

        b2 = tf.get_variable("obias", [1])

        h = tf.matmul(h, w2) + b2

        scope.reuse_variables()

    return h

yhat = model(x)

error = tf.nn.l2_loss(yhat-y)

train_step = tf.train.AdamOptimizer().minimize(error) #, var_list=[v1, v2, v3])





with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    log = MovingAverage(capacity=5)
    prevavg = 9e9 # large value

    untilresize = 7 
    begin = 0
    end = 100
    for i in xrange(100000):
        inarr, outarr = gensin(10000)
        sess.run(train_step, feed_dict={x:inarr, y:outarr, e: end, b: begin})

        if i % 1000 == 0:
            untilresize -= 1
            inarr, outarr = gensin(100)
            err = sess.run(error, feed_dict={x:inarr, y:outarr, e: end, b: begin})

            print err 
            log.addval(err)
            if log.getavg() >= prevavg and untilresize < 0:
                untilresize = 7
                prevavg = 9e9
                log = MovingAverage(capacity=5)
                if end < 500:
                    begin += 100
                    end += 100
            prevavg = log.getavg()
