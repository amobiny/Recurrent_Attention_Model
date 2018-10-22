"""
Defines several loss functions. They all follow the same API

    loss = loss_func(ground_truth, prediction, *kwargs)
"""

import tensorflow as tf

def location_loss(location_gt, location_hat, has_label):
    """ Mean squared error between ground truth central location and
    predicted location. Instances without label ('has_label'=0) are ignored.


    :param location_gt:     (B x S x 2) Tensor   central location ground truth
    :param location_hat:    (B x S x 2) Tensor   predicted location
    :param has_labels:      (B x 1) Tensor       boolean, has ground truth (=1) or not (=0)
    :return: loss           (1,) Tensor          mean squared error
    """
    assert location_gt.get_shape().as_list()==location_hat.get_shape().as_list(), '[location_loss]: location_gt must have same size location_hat.'

    # squared difference
    squared     = tf.square(location_gt - location_hat) # (B x S x 2)

    # sum over time steps and (x,y)
    mse         = tf.reduce_sum(squared, axis=[1,2])    # (B x 1)
    mse         = tf.expand_dims(mse,1)

    # ignore unlabeled data
    mse         = tf.multiply(mse, has_label)           # (B x 1)

    # average over instances
    return tf.reduce_mean(mse)

