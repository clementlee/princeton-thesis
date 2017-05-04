import tensorflow as tf

from tensorflow.python.training import moving_averages

# constants
FILTERS = [16, 32, 64, 128]
BLOCKS_PER = 10
NUM_LAYERS = 3 * BLOCKS_PER 
NUM_CLASSES = 100

# shortcuts
init = tf.contrib.layers.xavier_initializer

class Resnet:
    def __init__(self, phase):
        # placeholders
        #self.phase = tf.placeholder(tf.bool, name='phase')
        self.phase = phase

        self.cap = tf.placeholder(tf.float32, 
                shape=[NUM_LAYERS, 2], name="capacity")
        # self.cap = [[0, 0.1] for _ in xrange(NUM_LAYERS)]

        self._extra_train_ops = []

    def conv(self, name, x, idx, in_filters, out_filters,
            filter_size=3, first=False, stride=1):
        """Convolution method"""
        stride_arr = [1, stride, stride, 1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("kernel", 
                    [filter_size, filter_size, in_filters, out_filters],
                    initializer=init())

            # get capacities feeding in and going out
            if first:
                if idx > 0:
                    cin = self.cap[idx-1][1]
                else:
                    cin = 1.0
            else:
                cin = self.cap[idx][1]
            fout = self.cap[idx][0]
            cout = self.cap[idx][1]

            cin = tf.cast(cin * in_filters, tf.int32)
            fout = tf.cast(fout * out_filters, tf.int32)
            cout = tf.cast(cout * out_filters, tf.int32)

            # perform slices
            fkern = kernel[:,:,:cin,:fout]
            fkern = tf.stop_gradient(fkern)
            ckern = kernel[:,:,:cin,fout:cout]

            k = tf.concat([fkern,ckern], 3)

            return tf.nn.conv2d(x, k, stride_arr, padding='SAME')


    def batchnorm(self, name, x, size, c):
        # official documentation suggests against update_collections=None
        # but screw it this is easier

        #return tf.contrib.layers.batch_norm(x, activation_fn=tf.nn.elu,
        #        is_training=self.phase, center=True, scale=True,
        #        updates_collections=None)

        # use batch_norm, modified from tensorflow/models/resnet
        return self.batch_norm(name, x, size, c)

    def batch_norm(self, name, x, size, c):
        with tf.variable_scope(name):
            #params_shape = [x.get_shape()[-1]]
            params_shape = [size]
            
            beta = tf.get_variable(
	      	    'beta', params_shape, tf.float32,
	      	    initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
	      	    'gamma', params_shape, tf.float32,
	      	    initializer=tf.constant_initializer(1.0, tf.float32))

            n = tf.cast(size*c, tf.int32)
            beta = beta[:n]
            gamma = gamma[:n]
	
            if self.phase:
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
                
                pa = size - n
                mean = tf.pad(mean, [[0,pa]])
                variance = tf.pad(variance, [[0,pa]])
	
                moving_mean = tf.get_variable(
   	      	        'moving_mean', params_shape, tf.float32,
   	      	        initializer=tf.constant_initializer(0.0, tf.float32),
   	      	        trainable=False)
                moving_variance = tf.get_variable(
   	      	        'moving_variance', params_shape, tf.float32,
   	      	        initializer=tf.constant_initializer(1.0, tf.float32),
   	      	        trainable=False)

                #print moving_mean.get_shape()
                   
                #print moving_mean
                #moving_mean = moving_mean[:n]
                #print moving_mean
                #moving_variance = moving_variance[:n]
                
                self._extra_train_ops.append(
                           moving_averages.assign_moving_average(moving_mean, 
                               mean, 0.9))
                self._extra_train_ops.append(
                           moving_averages.assign_moving_average(moving_variance,
                               variance, 0.9))
            else:
   	      	    mean = tf.get_variable(
   	      	        'moving_mean', params_shape, tf.float32,
   	      	        initializer=tf.constant_initializer(0.0, tf.float32),
   	      	        trainable=False)
   	      	    variance = tf.get_variable(
   	      	        'moving_variance', params_shape, tf.float32,
   	      	        initializer=tf.constant_initializer(1.0, tf.float32),
   	      	        trainable=False)
   	      	    tf.summary.histogram(mean.op.name, mean)
   	      	    tf.summary.histogram(variance.op.name, variance)
   
               # epsilon = 0.001
            mean = mean[:n]
            variance = variance[:n]
            y = tf.nn.batch_normalization(
	      	    x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
        return y

    def resblock(self, x, idx, in_filters, out_filters, stride,
            before_activate = False):
        """residual block"""

        c1 = 1.0 if idx == 0 else self.cap[idx-1][1]
        if before_activate:
            with tf.variable_scope('before_activation'):
                x = self.batchnorm('init_bn', x, in_filters, c1)
                x = tf.nn.elu(x)
                orig_x = x
        else:
            with tf.variable_scope('after_activation'):
                orig_x = x
                x = self.batchnorm('init_bn', x, in_filters, c1)
                x = tf.nn.elu(x)

        with tf.variable_scope('sub1'):
            x = self.conv('conv1', x, idx, in_filters, out_filters, 
                    stride=stride, first=True)

        with tf.variable_scope('sub2'):
            x = self.batchnorm('bn2', x, out_filters, self.cap[idx][1])
            x = tf.nn.elu(x)
            x = self.conv('conv2', x, idx, out_filters, out_filters)

        with tf.variable_scope('sub_add'):
            # perform shortcut using convolutions
            if idx > 0:
                cin = self.cap[idx-1][1]
            else:
                cin = 1.0
            cout = self.cap[idx][1]

            if cin * in_filters != cout * out_filters:
                orig_x = self.conv('conv_proj', orig_x, idx, in_filters,
                        out_filters, filter_size=1, first=True, 
                        stride=stride)

            x += orig_x

        return x


    def model(self, x):
        # initial_conv
        with tf.variable_scope('init'):
            k = tf.get_variable('kernel', [3, 3, 3, FILTERS[0]], 
                    initializer=init())
            x =  tf.nn.conv2d(x, k, [1,1,1,1], padding='SAME')

        # sub 1
        with tf.variable_scope('unit_1_0'):
            x = self.resblock(x, 0, FILTERS[0], FILTERS[1], 1,
                    before_activate=True)
        for i in xrange(1, BLOCKS_PER):
            with tf.variable_scope('unit_1_%d' % i):
                x = self.resblock(x, i, FILTERS[1], FILTERS[1], 1)

        # sub 2
        with tf.variable_scope('unit_2_0'):
            x = self.resblock(x, BLOCKS_PER, FILTERS[1], FILTERS[2], 2)
        for i in xrange(1, BLOCKS_PER):
            with tf.variable_scope('unit_2_%d' % i):
                idx = i + BLOCKS_PER
                x = self.resblock(x, idx, FILTERS[2], FILTERS[2], 1)

        # sub 3
        with tf.variable_scope('unit_3_0'):
            x = self.resblock(x, 2 * BLOCKS_PER, FILTERS[2], FILTERS[3], 2)
        for i in xrange(1, BLOCKS_PER):
            with tf.variable_scope('unit_3_%d' % i):
                idx = i + 2 * BLOCKS_PER
                x = self.resblock(x, idx, FILTERS[3], FILTERS[3], 1)

        # final
        with tf.variable_scope('unit_last'):
            x = self.batchnorm('final_bn', x, FILTERS[3], self.cap[NUM_LAYERS-1][1])
            x = tf.nn.elu(x)
            x = tf.reduce_mean(x, [1,2])

        # fully-connected layer
        with tf.variable_scope('unit_fc'):
            fw = tf.get_variable('weights', [FILTERS[3], NUM_CLASSES],
                    initializer=init())
            b = tf.get_variable('biases', [NUM_CLASSES])
            cin = tf.cast(self.cap[3*BLOCKS_PER-1][1] * FILTERS[3], tf.int32)
            w = fw[:cin, :]

            x = tf.nn.xw_plus_b(x, w, b)

        return x



