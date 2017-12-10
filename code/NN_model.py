import tensorflow as tf

class ColorModel(object):

    def __init__(self, graph, l1_num_ofmaps, l2_num_ofmaps, lr):
        with graph.as_default():
            with tf.name_scope('conv_net'):

                self.im_inp = tf.placeholder(tf.float32, [None, None, None, None], 'input-image')
                self.im_outp = tf.placeholder(tf.float32, [None, None, None, None], 'output-image')

                # First conv layer
                filter_l1 = tf.random_uniform([3, 3, 1, l1_num_ofmaps], -1.0, 1.0)
                l1_b = tf.Variable(tf.zeros([l1_num_ofmaps]), name='l1_b')

                l1_conv = tf.nn.conv2d(self.im_inp, filter_l1, 1, "SAME", name='l1_conv')
                l1_conv_b = bias_add(l1_conv, l1_b, data_format='NHWC', name='l1_bias_add')
                l1_out = tf.nn.relu(l1_conv_b,  name='l1_act')

                # Second conv layer
                filter_l2 = tf.random_uniform([3, 3, l1_num_ofmaps, l2_num_ofmaps], -1.0, 1.0)
                l2_b = tf.Variable(tf.zeros([l2_num_ofmaps]), name='l2_b')

                l2_conv = tf.nn.conv2d(l1_out, filter_l2, 1, "SAME", name='l2_conv')
                l2_conv_b = bias_add(l2_conv, l2_b, data_format='NHWC', name='l2_bias_add')
                l2_out = tf.nn.relu(l2_conv_b,  name='l2_act')

                # Third conv layer
                filter_l3 = tf.random_uniform([3, 3, l2_num_ofmaps, 1], -1.0, 1.0)
                l3_b = tf.Variable(tf.zeros([1]), name='l3_b')

                l3_conv = tf.nn.conv2d(l2_out, filter_l3, 1, "SAME", name='l3_conv')
                l3_conv_b = bias_add(l3_conv, l3_b, data_format='NHWC', name='l3_bias_add')
                self.out = tf.nn.relu(l3_conv_b,  name='l3_act')

                self.loss = tf.nn.l2_loss(self.im_outp - self.out)
                learning_rate = tf.Variable(float(lr), trainable=False, dtype=tf.float32)
                self.global_step = tf.Variable(0, trainable=False, name='g_step')
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                                        self.loss, global_step=self.global_step)

    def step(self, session, im_inp, im_outp):
        """ Training step, returns the prediction, loss"""

        input_feed = {
            self.im_inp: im_inp,
            self.im_outp: im_outp,}

        output_feed = [self.loss, self.optimizer]
        return session.run(output_feed, input_feed)

    def dec_step(self, session, im_inp, im_outp):
        """ Training step, returns the prediction, loss"""

        input_feed = {
            self.im_inp: im_inp,}

        output_feed = [self.out]
        return session.run(output_feed, input_feed)
