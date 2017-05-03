import tensorflow as tf

init = tf.contrib.layers.xavier_initializer 


phase = tf.placeholder(tf.bool)
cap = [[0.0,1.0] for i in xrange(27)]
NUM_CLASSES = 100

def conv(name, filter_size, x, numin, numout, capacin, capacout, fixout, stride):
    fs = filter_size
    with tf.variable_scope(name):
        kernel = tf.get_variable('CW', [fs, fs, numin, numout], initializer=init())
        numfixed = int(fixout * numout)
        numtrain = int(capacout * numout)

        fixedin = int(capacin * numin)
        
        fix_kernel = kernel[:,:,:fixedin,:numfixed]
        fix_kernel = tf.stop_gradient(fix_kernel)

        train_kernel = kernel[:,:,:fixedin,numfixed:numtrain]

        full_kernel = tf.concat([fix_kernel, train_kernel], 3)
            
    return tf.nn.conv2d(x, full_kernel, [1, stride, stride, 1], padding='SAME')



# resblock
# adapted from google's tensorflow models https://github.com/tensorflow/models
def resblock(x, numin, numout, capacin, capacout, fixout, stride, activate_before = False):
    if activate_before:
        with tf.variable_scope("before_activate"):
            # x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
            x = tf.nn.elu(x)
            orig_x = x 
    else:
        with tf.variable_scope("residual_activation"):
            orig_x = x
            # x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
            x = tf.nn.elu(x)
    if capacout == 0:
        return x

    with tf.variable_scope("sub1"):
        x = conv('conv1', 3, x, numin, numout, capacin, capacout, fixout, stride)
    
    with tf.variable_scope("sub1"):
        # x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
        x = tf.nn.elu(x)
        x = conv('conv2', 3, x, numout, numout, capacout, capacout, fixout, stride)

    with tf.variable_scope("sub_add"):
        if numin != numout:
            sa = [1, stride, stride, 1]
            orig_x = tf.nn.avg_pool(orig_x, sa, sa, 'VALID')
            fixedin = int(capacin * numin)
            fixedout= int(capacout * numout)
            diff = (fixedout-fixedin)//2

            # have to fix off-by-one for slight difference in sizing
            if diff >= 0:
                if (fixedout-fixedin) %2 == 1:
                    orig_x = tf.pad(orig_x, [[0,0],[0,0],[0,0],[diff,diff+1]])
                else:
                    orig_x = tf.pad(orig_x, [[0,0],[0,0],[0,0],[diff,diff]])
        x += orig_x

    return x


# model
# capacities is array of pairs, specifying fixed and train capacities for each resblock
def model(x):
    with tf.variable_scope('init'):
        x = conv('init_conv', 3, x, 3, 16, 1.0, 1.0, 0.0, 1)


    #filters = [16, 160, 320, 640]
    filters = [16, 32, 64, 128]
    with tf.variable_scope('unit_1_0'):
        x = resblock(x, filters[0], filters[1], 1.0, cap[0][1], cap[0][0], 1, True)
    for i in xrange(1, 9):
        with tf.variable_scope('unit_1_%d' % i):
            x = resblock(x, filters[1], filters[1], cap[i-1][1], cap[i][1], cap[i][0], 1)

    with tf.variable_scope('unit_2_0'):
        x = resblock(x, filters[1], filters[2], cap[8][1], cap[9][1], cap[9][0], 1, True)
    for i in xrange(1, 9):
        with tf.variable_scope('unit_2_%d' % i):
            x = resblock(x, filters[2], filters[2], cap[i-1+9][1], cap[i+9][1], cap[i+18][0], 1)

    with tf.variable_scope('unit_3_0'):
        x = resblock(x, filters[2], filters[3], cap[17][1], cap[18][1], cap[18][0], 1, True)
    for i in xrange(1, 9):
        with tf.variable_scope('unit_3_%d' % i):
            x = resblock(x, filters[3], filters[3], cap[i-1+18][1], cap[i+18][1], cap[i+18][0], 1)

    with tf.variable_scope('unit_last'):
        # x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
        x = tf.nn.elu(x)
        x = tf.reduce_mean(x, [1,2])

    with tf.variable_scope('fc_layer'):
        # jprint x.get_shape()
        # jx = tf.reshape(x, [int(x.get_shape()[0]),-1]) # flatten with first channel = batch
        #x = tf.reshape(x, [10,-1]) # flatten with first channel = batch

        weights = tf.get_variable("weights", [filters[3], NUM_CLASSES], initializer=init())
        bias = tf.get_variable("bias", [NUM_CLASSES], initializer=tf.constant_initializer())

        small_capac = int(x.get_shape()[1])
        #small_capac *= 3 * 3
        sw = weights[:small_capac,:]

        return tf.nn.xw_plus_b(x, sw, bias)


#if __name__ == "__main__":
#    print "hi"
#    with tf.Session() as sess:
#        a = tf.get_variable("a", [10, 36, 36, 3])
#        b = model(a)
#        sess.run(tf.global_variables_initializer())
#        print sess.run(b, feed_dict={phase:False}).shape
# limit model

# 
