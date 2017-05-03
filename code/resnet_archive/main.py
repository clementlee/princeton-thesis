import tensorflow as tf
import numpy as np

import resnet 
import cifar_input
import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_root', 'checkpoints/', 'Where to store checkpoints.')
tf.app.flags.DEFINE_integer('batchsize', 100,  'batch size')
tf.app.flags.DEFINE_string('training_data', 'cifar-100-binary/train.bin', 'where training data is')
tf.app.flags.DEFINE_string('eval_data', 'cifar-100-binary/test.bin', 'where testing data is')
tf.app.flags.DEFINE_integer('num_gpus', 0,  'numbers of gpu to use (0 for none)')
tf.app.flags.DEFINE_string('mode', 'train',  'train or eval')

def main():
    x, y_ = cifar_input.build_input('cifar100', FLAGS.training_data, FLAGS.batchsize, FLAGS.mode)
    y_conv = resnet.model(x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.contrib.framework.get_or_create_global_step()
    
    with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.log_root, save_checkpoint_secs=5) as sess:
        # sess.run(tf.global_variables_initializer())

        # aa = sess.run(y_conv, feed_dict={resnet.phase: True})
        # ab = sess.run(y_)
        # print np.argmax(aa, axis=1)
        # print np.argmax(ab, axis=1)
        for i in xrange(100000):
            sess.run(train_step, feed_dict={resnet.phase: True})
            if i % 10 == 0:
                print sess.run(accuracy, feed_dict={resnet.phase: False})

if __name__ == '__main__':
    main()
