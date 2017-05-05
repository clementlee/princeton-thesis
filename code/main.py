import tensorflow as tf
import numpy as np

import cifar_input # using code from tensorflow/models/resnet

import resnet 

import time

from utilities import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, 'Training batch size')
tf.app.flags.DEFINE_string('checkpoints', 'checkpoints/', 'Checkpoint location')
tf.app.flags.DEFINE_string('training_data', 'cifar-100-binary/train.bin', 
        'Where to find training data')
tf.app.flags.DEFINE_string('eval_data', 'cifar-100-binary/test.bin', 
        'Where to find testing/evaluation data')
tf.app.flags.DEFINE_string('mode', 'train', 
        'either train or eval')


def train():
    images, labels = cifar_input.build_input('cifar100', FLAGS.training_data,
            FLAGS.batch_size, FLAGS.mode)

    phase = FLAGS.mode is 'train'
    model = resnet.Resnet(phase)
    tf.contrib.framework.get_or_create_global_step()
    y_conv = model.model(images)
    y_ = labels

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    ts = tf.train.AdamOptimizer().minimize(cross_entropy)
    tss = [ts] + model._extra_train_ops
    train_step = tf.group(*tss)


    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    #with tf.train.MonitoredTrainingSession(
    #        checkpoint_dir=FLAGS.checkpoints, save_checkpoint_secs=10,
    #        is_chief=True,
    #        config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        sess.run(tf.global_variables_initializer())

        path = tf.train.latest_checkpoint(FLAGS.checkpoints)
        if path:
            saver.restore(sess, path)
            print 'restoring from ' + path
        i = 0
        cap = np.array([[0,0.5] for _ in xrange(resnet.NUM_LAYERS)])


        log = MovingAverage()
        prevavg = 9e9 # large value

        RESIZE_MAX = 30
        untilresize = RESIZE_MAX
        increment = 0.2
        start = 0.0
        capac = increment
        cap = np.array([[start,capac] for _ in xrange(resnet.NUM_LAYERS)])
        while True:
            sess.run(train_step, feed_dict={model.cap:cap})
            if i % 100 == 0:
                a = sess.run([accuracy, cross_entropy], 
                        feed_dict={model.cap:cap})
               
                log.addval(a[0])
                if log.getavg() <= prevavg and untilresize < 0:
                    # resizing now
                    untilresize = RESIZE_MAX
                    prevavg = 9e9
                    log = MovingAverage()
                    if capac < 0.99:
                        start += increment
                        capac += increment
                        capac = min(1.0, capac)
                        cap = np.array([[start,capac] for _ in xrange(resnet.NUM_LAYERS)])
                    elif start > 0.5:
                        start -= increment
                        start = max(0.5, start)
                        cap = np.array([[start,capac] for _ in xrange(resnet.NUM_LAYERS)])
                prevavg = log.getavg()

                print "%f, %f, %f, %f" % (a[0], a[1], start, capac)
                saver.save(sess, FLAGS.checkpoints+'model.ckpt')
                np.save(FLAGS.checkpoints+'cap.npy', cap)
                untilresize -= 1
            i += 1


def eval():
    images, labels = cifar_input.build_input('cifar100', FLAGS.eval_data,
            FLAGS.batch_size, FLAGS.mode)
    cap = np.array([[0,1.0] for _ in xrange(resnet.NUM_LAYERS)])

    phase = FLAGS.mode is 'train'
    model = resnet.Resnet(phase)
    y_conv = model.model(images)

    saver = tf.train.Saver()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)


    total_prediction, correct_prediction = 0, 0

    best_precision = 0.0 
    while True:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoints)
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        cap = np.load(FLAGS.checkpoints+'cap.npy')
        for _ in xrange(10):
            (truth, predictions) = sess.run([labels,y_conv], feed_dict={model.cap:cap})
            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        print '%f, %f' % (precision, best_precision)

        time.sleep(60)


def example():
    q = tf.get_variable('asdf', [16, 30, 30, 3])
    asdf = resnet.Resnet()
    b = asdf.model(q)
    with tf.Session() as sess:
        cap[0,1] = 0.1
        sess.run(tf.global_variables_initializer())
        a = sess.run(b, feed_dict={asdf.cap: cap, asdf.phase:True})
        print a.shape

def main():
    if FLAGS.mode == 'train':
        train()
    else:
        with tf.device('/cpu:0'):
            eval()


if __name__ == '__main__':
    main()



