import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from data_generator import *

distributions = tf.contrib.distributions


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def loglikelihood(mean_arr, sampled_arr, sigma):
    mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
    sampled = tf.stack(sampled_arr)  # same shape as mu
    gaussian = distributions.Normal(mu, sigma)
    logll = gaussian._log_prob(sampled)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)  # sum over time steps
    logll = tf.transpose(logll)  # [batch_sz, timesteps]

    return logll


def conv2d(x,
           n_filters=16,
           kernel_size=(3, 3),
           stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           padding='SAME',
           name='conv2d'):
    """ Convolutional layer

  :param x              -- Tensorflow operator
  :param n_filters      -- (int) number of filters
  :param stride         -- (tuple) convolution stride (x,y)
  :param initializer    -- Tensorflow initializer
  :param activation_fn  -- Tensorflow activation function
  :param padding        -- padding mode
  :param name           -- variable scope
  """
    data_format = 'NHWC'  # height, width, input n_channels, output n_channels

    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], n_filters]

        # convolution operator
        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)
        b = tf.get_variable('biases', [n_filters], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

    if activation_fn != None:
        out = activation_fn(out)

    return out, w, b


def linear(x, out_dim, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    """ Linear layer

  :param x              -- Tensorflow operator
  :param out_dim        -- (int) # output connections
  :param stddev         -- (float) std for initialization
  :param bias_start     -- (float) bias constant for initialization
  :param activation_fn  -- Tensorflow activation function
  :param name           -- variable scope
  """
    shape = x.get_shape().as_list()

    with tf.variable_scope(name):

        w = tf.get_variable('weights', [shape[1], out_dim], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))

        b = tf.get_variable('bias', [out_dim],
                            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(x, w), b)

        if activation_fn != None:
            return activation_fn(out), w, b
        else:
            return out, w, b


def build_convnet(input_ph, layers, d_fc=None):
    """Builds convet given a dictionary containing hyper-parameters.

    Args:
        x         -- input Tensor
        layers    -- (dict) key -- values are
                    layer_nr (int): [filter_w, n_filters]
        d_fc    -- if int, adds linear layer with 'd_fc' units

    Returns:
        output    -- (operator) output vector
        weights   -- (dict) layer_nr: weight
        biases    -- (dict) layer_nr: bias
    """
    n_layers = len(layers.keys())
    print '\n\n[building convnet]: {} layers, {} units in fully-connected\n'.format(n_layers, d_fc)

    w, b = {}, {}

    h = input_ph

    for l in range(n_layers):
        print 'Layer {} -- {} filters, {}x{}'.format(l, layers[l][0], layers[l][1], layers[l][1])
        h, w[l], b[l] = conv2d(x=h,
                               n_filters=layers[l][0],
                               kernel_size=(layers[l][1], layers[l][1]),
                               name='conv{}'.format(l)
                               )
        h = maxpool2d(x=h, k=2)

    if d_fc != None:
        shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, reduce(lambda x, y: x * y, shape[1:])], name='flatten')  # flatten
        h, w[n_layers], b[n_layers] = linear(h, out_dim=d_fc, activation_fn=tf.nn.relu, name='fc1')
        return h, w, b
    else:
        return h, w, b


# evaluation
def evaluate(sess, images_ph, labels_ph, softmax, mnist, config, task):
    """Return current test and validation accuracy."""

    print 'Evaluating on {} task ({}x{}, {} distractors) using {} glimpses (at {} scales)'.format(
        task, config.new_size, config.new_size, config.n_distractors,
        config.num_glimpses, config.n_patches)

    # Evaluation
    test_acc = []
    val_acc = []

    for k, dataset in enumerate([mnist.validation, mnist.test]):

        steps_per_epoch = dataset.num_examples // config.eval_batch_size
        correct_cnt = 0
        num_samples = steps_per_epoch * config.batch_size
        # loc_net.sampling = True

        for test_step in tqdm(xrange(steps_per_epoch)):

            images, labels = dataset.next_batch(config.batch_size)
            images = images.reshape((-1, config.original_size, config.original_size, 1))
            labels_bak = labels

            if task == 'translated':
                images = translate(images, width=config.new_size, height=config.new_size)
            elif task == 'cluttered':
                images = clutter(images,
                                 dataset.images.reshape((-1, config.original_size, config.original_size, 1)),
                                 width=config.new_size, height=config.new_size, n_patches=config.n_distractors
                                 )
            elif task == 'cluttered_var':
                images, _, _, _ = clutter_rnd(images,
                                              train_data=dataset.images.reshape(
                                                  (-1, config.original_size, config.original_size, 1)),
                                              lim=config.distractor_range,
                                              color_digits=config.color_digits,
                                              color_noise=config.color_noise,
                                              width=config.new_size, height=config.new_size, norm=True)

            # else:
            #    print 'original mnist data ({}x{}).'.format(config.original_size,config.original_size)

            # Duplicate M times (average prediction over M repeats)
            images = np.tile(images, [config.M, 1, 1, 1])
            labels = np.tile(labels, [config.M])

            softmax_val = sess.run(softmax,
                                   feed_dict={
                                       images_ph: images,
                                       labels_ph: labels
                                   })
            softmax_val = np.reshape(softmax_val,
                                     [config.M, -1, config.num_classes])
            softmax_val = np.mean(softmax_val, 0)

            pred_labels_val = np.argmax(softmax_val, 1)
            correct_cnt += np.sum(pred_labels_val == labels_bak)
        acc = correct_cnt / float(num_samples)

        if k == 0:
            print '\nVal accuracy\t{:4.4f} ({:4.4f} error)'.format(100 * acc, 100 - 100 * acc)
            val_acc = acc
        else:
            print 'Test accuracy\t{:4.4f} ({:4.4f} error)\n'.format(100 * acc, 100 - 100 * acc)
            test_acc = acc

    return test_acc, val_acc


def evaluate_loc(sess, images_ph, labels_ph, locations_ph, has_label, softmax, mnist, config, task):
    """Returns current test and validation accuracy."""

    if config.color_digits or config.color_noise:
        print 'Evaluating on {} task -- {}x{}, color digits: {}, color noise: {}\n' \
              'Model: {} patches, {} glimpses, glimpse size {}x{}\n\n\n'.format(
            task, config.new_size, config.new_size, config.color_digits, config.color_noise,
            config.n_patches, config.num_glimpses, config.glimpse_size, config.glimpse_size
        )
    else:
        print 'Evaluating on {} task ({}x{}, {} distractors) using {} glimpses (at {} scales)'.format(
            task, config.new_size, config.new_size, config.n_distractors,
            config.num_glimpses, config.n_patches)

    # Evaluation
    test_acc = []
    val_acc = []

    # for k, dataset in enumerate([mnist.validation, mnist.test]):
    for k, dataset in enumerate([mnist.test]):

        steps_per_epoch = dataset.num_examples // config.eval_batch_size
        correct_cnt = 0
        num_samples = steps_per_epoch * config.batch_size
        # loc_net.sampling = True

        for test_step in tqdm(xrange(steps_per_epoch)):

            images, labels = dataset.next_batch(config.batch_size)
            images = images.reshape((-1, config.original_size, config.original_size, 1))
            labels_bak = labels

            if task == 'translated':
                images, locations, _ = translate_loc(images, width=config.new_size, height=config.new_size, norm=True)
            elif task == 'cluttered':
                images, locations, _ = clutter_loc(images,
                                                   dataset.images.reshape(
                                                       (-1, config.original_size, config.original_size, 1)),
                                                   width=config.new_size, height=config.new_size,
                                                   n_patches=config.n_distractors,
                                                   norm=True)
            elif task == 'cluttered_var':
                images, locations, bboxes, nd = clutter_rnd(images,
                                                            train_data=dataset.images.reshape(
                                                                (-1, config.original_size, config.original_size, 1)),
                                                            lim=config.distractor_range,
                                                            color_digits=config.color_digits,
                                                            color_noise=config.color_noise,
                                                            width=config.new_size, height=config.new_size, norm=True)

            # mask out subset of labels
            has_labels = (np.random.rand(config.batch_size) < config.p_labels).astype(np.int32)

            # Duplicate M times (average prediction over M repeats)
            images = np.tile(images, [config.M, 1, 1, 1])
            labels = np.tile(labels, [config.M])
            locations = np.tile(locations, [config.M, 1])
            has_labels = np.tile(has_labels, [config.M])

            softmax_val = sess.run(softmax,
                                   feed_dict={
                                       images_ph: images,
                                       labels_ph: labels,
                                       locations_ph: locations,
                                       has_label: has_labels,
                                   })
            softmax_val = np.reshape(softmax_val,
                                     [config.M, -1, config.num_classes])
            softmax_val = np.mean(softmax_val, 0)

            pred_labels_val = np.argmax(softmax_val, 1)
            correct_cnt += np.sum(pred_labels_val == labels_bak)
        acc = correct_cnt / float(num_samples)

        if k == 0:
            print 'Test accuracy\t{:4.4f} ({:4.4f} error)\n'.format(100 * acc, 100 - 100 * acc)
            test_acc = acc
            val_acc = 0.0
        else:
            print 'Val accuracy\t{:4.4f} ({:4.4f} error)\n'.format(100 * acc, 100 - 100 * acc)
            test_acc = acc

    return test_acc, val_acc


def evaluate_repeatedly(ram=None, data=[], task='translated', n_reps=5, verbose=True):
    """Repeats trained RAM model (n_reps) times and returs accuracies and errors.

    Args:
        ram         (RAM object)
        data        (data object) same structure as TF mnist
        task        (str) evaluation test
        n_reps      (int) # repetitions
        verbose     (bool) print results or not

    Returns:
        accuracies  (list) accuracies in [0,1]
        errors      (list) errors in [0,1] (1-accuracy)
    """
    accuracies = {'test': [], 'val': []}
    errors = {'test': [], 'val': []}

    for k in range(n_reps):
        test, val = ram.evaluate(data=data, task=task)

        accuracies['test'].append(100 * test)
        errors['test'].append(100 - 100 * test)

        accuracies['val'].append(100 * val)
        errors['val'].append(100 - 100 * val)

    if verbose:
        print '\b-- Results ({} reps) ..\nTest:\t{:4.4f} +- {:4.4f} (error {:4.4f} +- {:4.4f})\n' \
              'Val:\t{:4.4f} +- {:4.4f} (error {:4.4f} +- {:4.4f})\n'.format(
            n_reps,
            np.mean(accuracies['test']), np.std(accuracies['test']),
            np.mean(errors['test']), np.std(errors['test']),
            np.mean(accuracies['val']), np.std(accuracies['val']),
            np.mean(errors['val']), np.std(errors['val'])
        )

    return accuracies, errors


def location_bounds(glimpse_w, input_w):
    """Given input image width and glimpse width returns (lower,upper) bound in (-1,1) for glimpse centers.

    :param: int  glimpse_w      width of glimpse patch
    :param: int  input_w        width of input image

    :return: int lower          lower bound in (-1,1) for glimpse center locations
    :return: int upper
    """
    offset = float(glimpse_w) / input_w
    lower = (-1 + offset)
    upper = (1 - offset)

    assert lower >= -1 and lower <= 1, 'lower must be in (-1,1), is {}'.format(lower)
    assert upper >= -1 and upper <= 1, 'upper must be in (-1,1), is {}'.format(upper)

    return lower, upper


def norm2ind(norm_ind, width):
    """Converts from (-1,1) to pixel indices.

    :param norm_ind     2D array containing (y,x) coordinates in (-1,1)
    :param width        image width
    """
    return np.round(width * ((norm_ind + 1) / 2.0), 1)


def ind2norm(ind, width):
    """Converts from pixel indices to (-1,1).

    :param ind          2D array containing (y,x) coordinates as pixel indices
    :param width        image width
    """
    return np.round((ind / float(width)) * 2 - 1, 1)


def count_parameters(sess):
    """Returns the number of parameters of a computational graph."""

    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    n_params = 0

    for k, v in zip(variables_names, values):
        print '-'.center(140, '-')
        print '{:60s}\t\tShape: {:20s}\t{:20} parameters'.format(k, v.shape, v.size)

        n_params += v.size

    print '-'.center(140, '-')
    print 'Total # parameters:\t\t{}\n\n'.format(n_params)

    return n_params


def sample_eval(config, data, n_eval=64):
    eval_idx = np.random.permutation(5000)[:n_eval]
    val_images, val_labels = data.validation.next_batch(5000)
    val_images, val_labels = val_images[eval_idx], val_labels[eval_idx]
    val_images = val_images.reshape((-1, config.original_size, config.original_size, 1))

    val_images, val_locations, _, _ = clutter_rnd(val_images,
                                                  train_data=data.validation.images.reshape(
                                                      (-1, config.original_size, config.original_size, 1)),
                                                  lim=config.distractor_range,
                                                  color_digits=config.color_digits,
                                                  color_noise=config.color_noise,
                                                  width=config.new_size, height=config.new_size, norm=True)
    val_has_labels = np.ones((n_eval))
    val_images = np.tile(val_images, [config.M, 1, 1, 1])
    val_labels = np.tile(val_labels, [config.M])
    val_locations = np.tile(val_locations, [config.M, 1])
    has_labels = np.tile(val_has_labels, [config.M])

    return val_images, val_labels, val_locations, val_has_labels
