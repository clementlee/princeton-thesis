
import tensorflow as tf

# from https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
def prelu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

# resblock
# adapted from google's tensorflow models https://github.com/tensorflow/models
def resblock(x, numin, numout, capacity, stride, phase):
    with tf.variable_scope("before_activate"):
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
        x = prelu(x)
        orig_x = x 

    with tf.variable_scope("sub1"):


# model

# limit model

# 
