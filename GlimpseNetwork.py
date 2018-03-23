"""
    Implements the Glimpse Network.

    This module  extracts a feature vector g_t (at time t)
    for a given image x_t and a glimpse location l_(t-1).
    The feature vector is the input to the LSTM.

    Structure:
        Two parts:
            1. Glimpse sensor: produces k retinal patches (R^{image} -> R^{batch_size} x R^{bandwidth})
            2. "Location encoder": maps from l_(t-1) to feature vector (R^{2} -> R^{hidden})

        The outputs of 1 and 2 are then combined into output vector g_t

    Input:
        x_t         (4D tensor) input image(s)
        l_(t-1)     (tuple) real-valued (x,y) coordinates with (0,0) being the center and (-1,-1) top-left corner

    Output:
        g_t         (2D tensor) feature vector

Partly inspired by https://github.com/zhongwen/RAM/blob/master/glimpse.py
"""

import numpy        as np
import tensorflow   as tf
from config import Config
from src.utils import *


class GlimpseNetwork(object):
    """ Takes image and previous glimpse location and outputs feature vector."""

    def __init__(self, config, images_ph):
        """
        :param config:      (object) hyperparams
        :param images_ph:   (placeholder) 4D tensor, format 'NHWC'
        """

        # get input dimensions
        N, W, H, C = images_ph.get_shape().as_list()

        self.original_size = W  # assume square
        self.num_channels = C  # input channels
        self.sensor_size = config.sensor_size  # dim. when glimpses are concatenated
        self.n_patches = config.n_patches  # glimpses at each t
        self.glimpse_size = config.glimpse_size  # width (=height) of each glimpse
        self.scale = config.scale  # how much receptive fields increase
        self.convert_ratio = config.convert_ratio  # ratio converting unit width to pixels

        self.minRadius = config.minRadius
        self.hg_size = config.hg_size  # hidden feature vector glimpse network
        self.hl_size = config.hl_size  # hidden feature vector 'location encoder'
        self.g_size = config.g_size  # output dimensionality
        self.loc_dim = config.loc_dim  # dim. location (=2)

        self.images_ph = images_ph  # image placeholder (NHWC)

        self.init_weights()

    def init_weights(self):
        """ Initialize all trainable weights."""

        # image -> vector hg
        with tf.variable_scope('glimpse_sensor'):
            self.w_g0 = weight_variable((self.sensor_size, self.hg_size))
            self.b_g0 = bias_variable((self.hg_size,))

            # linear(hg)
            self.w_g1 = weight_variable((self.hg_size, self.g_size))
            self.b_g1 = bias_variable((self.g_size,))

        # location -> vector hl
        with tf.variable_scope('location_encoder'):
            self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
            self.b_l0 = bias_variable((self.hl_size,))

            # linear(hl)
            self.w_l1 = weight_variable((self.hl_size, self.g_size))
            self.b_l1 = bias_variable((self.g_size,))

    def get_glimpse(self, loc):
        """Take glimpse on the original images.

        :return: tensor
        """
        loc = tf.stop_gradient(loc)

        # extract k glimpses (N x W x H x C)
        # each glimpse covers a larger field
        glimpse_size = self.glimpse_size
        glimpses = []
        glimpse_img = []

        for glimpseI in range(self.n_patches):
            # glimpse

            # make sure glimpses are within image
            lower, upper = location_bounds(glimpse_size, self.original_size)
            loc = tf.clip_by_value(loc, lower, upper)

            # print '[debug:] Glimpse {}, size {}, bounds {}'.format(glimpseI, glimpse_size, (lower,upper))

            glimpse = tf.image.extract_glimpse(self.images_ph,
                                               [glimpse_size, glimpse_size],
                                               loc,
                                               centered=True, normalized=True,  # (-1,1): upper-left, (0,0): center
                                               name='extract_glimpse')

            # crop to standard size
            glimpse = tf.image.resize_bilinear(glimpse, [self.glimpse_size, self.glimpse_size], name='resize')

            # print '--[glimpse net:] glimpse {}'.format(glimpse.get_shape().as_list())

            # flatten: N x (W^2*C*n_glimpses)
            glimpses.append(
                tf.reshape(glimpse, [-1, self.glimpse_size ** 2 * self.num_channels], name='reshape')
            )

            glimpse_img.append(glimpse)

            # scale glimpse
            glimpse_size *= self.scale

        # concatenate all glimpses
        glimpses = tf.stack(glimpses)  # n_patches x N x W^2 * C

        # reshape to batch size x (W^2*C*n_patches)
        glimpses = tf.transpose(glimpses, [1, 0, 2])  # N x n_patches x W^2 * C
        k, d = glimpses.get_shape().as_list()[1:]
        glimpses = tf.stop_gradient(tf.reshape(glimpses, [-1, k * d]))  # N x n_patches * W^2 * C
        # print '[glimpse net:] glimpses {}'.format(glimpses.get_shape().as_list())

        # for visualization
        self.glimpse_img = tf.stack(glimpse_img, axis=1)  # (B x n_patches x H x W x C)
        # print '\n[glimpse net:] self.glimpses_img {}'.format(self.glimpse_img.get_shape().as_list())

        return glimpses

    def __call__(self, loc):
        """ Returns feature vector g_t upon call, e.g.

        >> gn       = GlimpseNetwork(config, img)
        >> glimpse  = gn(loc)

        :param loc      -- [batch_size x 2] Tensor

        """
        # print '[glimpse_net:] loc dim {}'.format(loc.get_shape().as_list())

        # flattened glimpse patches
        glimpse_input = self.get_glimpse(loc)
        # glimpse_input = tf.reshape(glimpse_input,
        #                       (tf.shape(loc)[0], self.sensor_size)) # ensure right input format

        # print '[glimpse_net:] glimpses dim {}'.format(glimpse_input.get_shape().as_list())

        # glimpse sensor
        g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
        g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)  # Linear(h_g)

        # 'location encoder'
        l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
        l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)  # Linear(h_l)

        # combine
        g = tf.nn.relu(g + l)
        return g


class LocNet(object):
    """ Location network.
      Takes RNN output and produces the 2D mean for next location which
      is then sampled from a 2D Gaussian with mean and fixed variance.
  """

    def __init__(self, config):
        self.loc_dim = config.loc_dim
        self.input_dim = config.cell_output_size
        self.loc_std = config.loc_std
        self.convert_ratio = config.convert_ratio
        self._sampling = True  # boolean: sample or not

        self.init_weights()

    def init_weights(self):
        self.w = weight_variable((self.input_dim, self.loc_dim))
        self.b = bias_variable((self.loc_dim,))

    def __call__(self, input):
        """Takes RNN output and predicts next location.

    :param input        -- (cell_output_size) Tensor
    :return loc, mean   -- sampled location, mean of distribution
    """
        mean = tf.tanh(tf.nn.xw_plus_b(input, self.w, self.b))  # # in [-1, 1]

        # control mapping from (-1,1) to pixels
        if self.convert_ratio != None:
            mean = mean * self.convert_ratio

        if self._sampling:
            loc = mean + tf.random_normal((tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
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
