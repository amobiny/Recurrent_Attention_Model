"""
    Implements the Convolutional Glimpse Network described in Ba et al 2015.

    This module extracts a feature vector g_t (at time t)
    for a given image x_t and a glimpse location l_(t-1).

    Two components

    1. Glimpse sensor ("what"):     image -> 3 x convolutional layers -> fully-connected -> hg
    2. Location encoder ("where"):  (lx,ly) -> hl

    Different to Mnih et al 2014 the vectors and then multiplied element-wise.

    Partly inspired by https://github.com/zhongwen/RAM/blob/master/glimpse.py
"""

import  numpy        as np
import  tensorflow   as tf
from    src.utils        import *

class GlimpseNetwork(object):
    """ Takes image and previous glimpse location and outputs feature vector."""

    def __init__(self, config, images_ph):
        """
        :param config:      (object) hyperparams
        :param images_ph:   (placeholder) 4D tensor, format 'NHWC'
        """

        # get input dimensions
        N, W, H, C          = images_ph.get_shape().as_list()

        self.conv_layers    = config.conv_layers
        self.original_size  = config.new_size      # assume square
        self.num_channels   = C                    # input channels
        self.sensor_size    = config.sensor_size   # dim. when glimpses are concatenated
        self.n_patches      = config.n_patches     # glimpses at each t
        self.glimpse_size   = config.glimpse_size  # width (=height) of each glimpse
        self.scale          = config.scale         # how much receptive fields increase

        self.minRadius      = config.minRadius
        self.hg_size        = config.hg_size       # hidden feature vector glimpse network
        self.hl_size        = config.hl_size       # hidden feature vector 'location encoder'
        self.g_size         = config.g_size        # output dimensionality
        self.loc_dim        = config.loc_dim       # dim. location (=2)

        self.images_ph      = images_ph            # image placeholder (NHWC)

        self.init_weights()

    def init_weights(self):
        """ Initialize all trainable weights."""

        with tf.variable_scope('glimpse_sensor'):
            with tf.variable_scope('convolutions'):
                self.w_g0 = tf.get_variable("conv1", shape=[3,3,3,32],initializer=tf.contrib.layers.xavier_initializer())
                self.b_g0 = tf.Variable(tf.constant(0.1,shape=[32]),name='b_conv1')

                self.w_g1 = tf.get_variable("conv2", shape=[1,1,32,32],initializer=tf.contrib.layers.xavier_initializer())
                self.b_g1 = tf.Variable(tf.constant(0.1,shape=[32]),name='b_conv2')

                #self.w_g2 = tf.get_variable("conv3", shape=[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer())
                #self.b_g2 = tf.Variable(tf.constant(0.1,shape=[256]),name='b_conv2')

            # fully connected
            with tf.variable_scope('fully_connected'):
                n_units = (self.glimpse_size/2)**2 * 32 # 3 x max pooling
                self.w_g3 = weight_variable((n_units, self.g_size), name='weights')
                self.b_g3 = bias_variable((self.g_size,), name='bias')

                # concate -> hidden vector
                with tf.variable_scope('combine'):
                    self.w_g4 = weight_variable((self.n_patches * self.g_size, self.g_size), name='weights')
                    self.b_g4 = bias_variable((self.g_size,), name='bias')

        #location -> vector hl
        with tf.variable_scope('location_encoder'):
            self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
            self.b_l0 = bias_variable((self.hl_size,))

        #linear(hl)
        with tf.variable_scope('combined_where_and_what'):
            self.w_l1 = weight_variable((self.hl_size, self.g_size))
            self.b_l1 = bias_variable((self.g_size,))

    def get_glimpse(self, loc):
        """Take glimpse on the original images.

        :return: tensor
        """
        loc = tf.stop_gradient(loc)

        # extract k glimpses (N x W x H x C)
        # each glimpse covers a larger field
        glimpse_size    = self.glimpse_size
        glimpses        = []
        glimpse_img     = []

        for glimpseI in range(self.n_patches):

            # glimpse

            # make sure glimpses are within image
            lower, upper    = location_bounds(glimpse_size, self.original_size)
            loc             = tf.clip_by_value(loc,lower,upper)

            glimpse = tf.image.extract_glimpse(self.images_ph,
                                                [glimpse_size, glimpse_size],
                                                loc,
                                                centered=True, normalized=True, # (-1,1): upper-left, (0,0): center
                                                name='extract_glimpse')

            # crop to standard size
            glimpse = tf.image.resize_bilinear(glimpse, [self.glimpse_size, self.glimpse_size], name='resize')

            # flatten: N x (W^2*C*n_glimpses)
            glimpses.append(
                    tf.reshape(glimpse, [-1, self.glimpse_size**2 * self.num_channels],name='reshape')
                    )

            glimpse_img.append(glimpse)

            # scale glimpse
            glimpse_size *= self.scale

        # concatenate all glimpses
        glimpses    = tf.stack(glimpses)  # n_patches x N x W^2 * C

        # reshape to batch size x (W^2*C*n_patches)
        glimpses    = tf.transpose(glimpses,[1,0,2]) # N x n_patches x W^2 * C
        k,d         = glimpses.get_shape().as_list()[1:]
        glimpses    = tf.stop_gradient(tf.reshape(glimpses,[-1,k*d])) # N x n_patches * W^2 * C

        # for visualization
        self.glimpse_img = tf.stack(glimpse_img, axis=1) # (B x n_patches x H x W x C)

        return self.glimpse_img


    def __call__(self, loc):
        """ Returns feature vector g_t upon call, e.g.

        >> gn       = GlimpseNetwork(config, img)
        >> glimpse  = gn(loc)

        """

        # flattened glimpse patches
        glimpse_input = self.get_glimpse(loc) # (N x P x W x H x C) P: # patches

        # get individual glimpses
        patches = tf.unstack(glimpse_input, axis=1)

        # concatenate convolutional features for each patch
        hg = []

        for i in range(self.n_patches):

            # glimpse sensor
            strides = [1,1,1,1]
            h = tf.nn.relu(tf.nn.conv2d(patches[i], self.w_g0, strides=strides, padding='SAME') + self.b_g0)
            h = maxpool2d(h, k=2)
            # 24 x 24

            h = tf.nn.relu(tf.nn.conv2d(h, self.w_g1, strides=strides, padding='SAME') + self.b_g1)
            #h = maxpool2d(h, k=2)
            # 12 x 12

            #h = tf.nn.relu(tf.nn.conv2d(h, self.w_g2, strides=strides, padding='SAME') + self.b_g2)
            #h = maxpool2d(h, k=2)
            # 6 x 6

            h = tf.contrib.layers.flatten(h)
            h = tf.nn.relu(tf.nn.xw_plus_b(h, self.w_g3, self.b_g3))
            hg.append(h)

        hg = tf.stack(hg, axis=1)

        hg = tf.contrib.layers.flatten(hg)
        hg = tf.nn.relu(tf.nn.xw_plus_b(hg, self.w_g4, self.b_g4))

        # 'location encoder'
        l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
        l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1) # Linear(h_l)

        # combine
        g = tf.nn.relu(hg + l)

        return g

class LocNet(object):
  """ Location network.
      Takes RNN output and produces the 2D mean for next location which
      is then sampled from a 2D Gaussian with mean and fixed variance.
  """

  def __init__(self, config):
    self.loc_dim        = config.loc_dim
    self.input_dim      = config.cell_output_size
    self.loc_std        = config.loc_std
    self.convert_ratio  = config.convert_ratio
    self._sampling      = True              # boolean: sample or not

    self.init_weights()

  def init_weights(self):
    self.w              = weight_variable((self.input_dim, self.loc_dim))
    self.b              = bias_variable((self.loc_dim,))

  def __call__(self, input):
    """Takes RNN output and predicts next location.

    :param input        -- (cell_output_size) Tensor
    :return loc, mean   -- sampled location, mean of distribution
    """
    mean                = tf.tanh(tf.nn.xw_plus_b(input, self.w, self.b)) # # in [-1, 1]

    # control mapping from (-1,1) to pixels
    if self.convert_ratio != None:
        mean = mean * self.convert_ratio

    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean

    loc = tf.stop_gradient(loc)

    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling
