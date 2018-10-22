"""
Functions to generate, on the fly

    1. translated MNIST
    2. cluttered MNIST

The default parameters, such as the background size are taken from [1].

"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt
from src.fig import create_bbox

def translate(batch, width=60, height=60):
    """Inserts MNIST digits at random locations in larger blank background."""

    n, width_img, height_img, c_img = batch.shape

    data    = np.zeros((n, width, height, c_img)) # blank background for each image

    for k in range(n):

        # sample location
        x_pos   = np.random.randint(0,width - width_img)
        y_pos   = np.random.randint(0,height - height_img)

        # insert in blank image
        data[k, x_pos:x_pos+width_img, y_pos:y_pos+height_img, :] += batch[k]

    return data


def clutter(batch, train_data, width=60, height=60, n_patches=4):
    """Inserts MNIST digits at random locations in larger blank background and
    adds 8 by 8 subpatches from other random MNIST digits."""

    # get dimensions
    n, width_img, height_img, c_img = batch.shape
    width_sub, height_sub           = 8,8 # subpatch

    assert n > 4, 'There must be more than 4 images in the batch (there are {})'.format(n)

    data    = np.zeros((n, width, height, c_img)) # blank background for each image

    for k in range(n):

        # sample location
        x_pos   = np.random.randint(0,width - width_img)
        y_pos   = np.random.randint(0,height - height_img)

        # insert in blank image
        data[k, x_pos:x_pos+width_img, y_pos:y_pos+height_img, :] += batch[k]

        # add 8 x 8 subpatches from random other digits
        for i in range(n_patches):
            digit   = train_data[np.random.randint(0, train_data.shape[0]-1)]
            c1, c2  = np.random.randint(0, width_img - width_sub, size=2)
            i1, i2  = np.random.randint(0, width - width_sub, size=2)
            data[k, i1:i1+width_sub, i2:i2+height_sub, :] += digit[c1:c1+width_sub, c2:c2+height_sub, :]

    data = np.clip(data, 0., 1.)

    return data

# data sets including location ground truth (bbox & central location tuple)
def translate_loc(batch, width=60, height=60, norm=True):
    """Inserts MNIST digits at random locations in larger blank background.

    Returns images and ground truth locations and ground truth bounding boxes (x,y,x_width,y_width)

    If norm=True, locations/bboxes are given in (-1,1)
    """

    n, width_img, height_img, c_img = batch.shape

    halfwidth_img   = width_img/2.0
    data            = np.zeros((n, width, height, c_img)) # blank background for each image

    bboxes          = np.empty((n, 4))
    locations       = np.empty((n, 2))

    for k in range(n):

        # sample location
        x_pos   = np.random.randint(0, width - width_img)
        y_pos   = np.random.randint(0, height - height_img)

        # store locations
        # central location, format [-1,1]
        if norm:
            bboxes[k,:]         = (x_pos/float(width))*2.0 - 1.0, (y_pos/float(width))*2.0 - 1.0, width_img, width_img
            locations[k,:]      = x_pos+halfwidth_img, y_pos+halfwidth_img
            locations[k,:]      = (locations[k,:]/float(width))*2.0 - 1.0
        else:
            bboxes[k,:]         = x_pos, y_pos, width_img, width_img
            locations[k,:]      = x_pos+halfwidth_img, y_pos+halfwidth_img

        # insert in blank image
        data[k, x_pos:x_pos+width_img, y_pos:y_pos+height_img, :] += batch[k]

    return data, locations, bboxes

def clutter_loc(batch, train_data, width=60, height=60, n_patches=4, norm=True):
    """Inserts MNIST digits at random locations in larger blank background and
    adds 8 by 8 subpatches from other random MNIST digits.

    Returns images and ground truth locations and ground truth bounding boxes (x,y,x_width,y_width)

    If norm=True, locations/bboxes are given in (-1,1)
    """

    # get dimensions
    n, width_img, height_img, c_img = batch.shape
    width_sub, height_sub           = 8,8 # subpatch
    halfwidth_img                   = width_img/2.0

    assert n >= 4, 'There must be more than 4 images in the batch (there are {})'.format(n)

    data    = np.zeros((n, width, height, c_img)) # blank background for each image

    bboxes          = np.empty((n, 4))
    locations       = np.empty((n, 2))
    locations_norm  = np.empty((n, 2))

    for k in range(n):

        # sample location
        x_pos   = np.random.randint(0,width - width_img)
        y_pos   = np.random.randint(0,height - height_img)

        # store locations
        # central location, format [-1,1]
        if norm:
            bboxes[k,:]         = (x_pos/float(width))*2.0 - 1.0, (y_pos/float(width))*2.0 - 1.0, width_img, width_img
            locations[k,:]      = x_pos+halfwidth_img, y_pos+halfwidth_img
            locations[k,:]      = (locations[k,:]/float(width))*2.0 - 1.0
        else:
            bboxes[k,:]         = x_pos, y_pos, width_img, width_img
            locations[k,:]      = x_pos+halfwidth_img, y_pos+halfwidth_img

        # insert in blank image
        data[k, x_pos:x_pos+width_img, y_pos:y_pos+height_img, :] += batch[k]

        # add 8 x 8 subpatches from random other digits
        for i in range(n_patches):
            digit   = train_data[np.random.randint(0, train_data.shape[0]-1)]
            c1, c2  = np.random.randint(0, width_img - width_sub, size=2)
            i1, i2  = np.random.randint(0, width - width_sub, size=2)
            data[k, i1:i1+width_sub, i2:i2+height_sub, :] += digit[c1:c1+width_sub, c2:c2+height_sub, :]

    data = np.clip(data, 0., 1.)

    return data, locations, bboxes

def clutter_rnd(batch, train_data, lim=(4,14), color_digits=False, color_noise=False, width=60, height=60, norm=True):
    """Generates cluttered MNIST data.

    Method:
        MNIST digits are inserted into a large background and random (8 x 8) patches from other
        training examples are added as noise ('distractors'). The number of such patches is different for every
        generated example.

        The digits and the distractors can be in color or grayscale.

    Args:
        batch (np.array):       (B x W x H x C) batch MNIST images
        train_data (np.array):  (N x W x H x C) all MNIST images (to sampel distractors)
        lim (tuple):            (lower, upper) bound for number of distractors
        color_digits (bool):    True: random color for digit, False: white
        color_noise (bool):     True: random colors for distractors, False: white
        width (int):            background width
        height(int):            background height
        norm (bool):            True: bound boxes/locations in (-1,1), False: (0,width/height)

    Returns:
        X (np.array):          (B x width x height x C) cluttered MNIST digits
        locations (np.array):  (B x width x height x C) cluttered MNIST digits
    """

    # get dimensions
    n, width_img, height_img, c_img = batch.shape
    width_sub, height_sub           = 8, 8 # subpatch
    halfwidth_img                   = width_img/2.0
    halfheight_img                  = height_img/2.0

    if color_digits or color_noise:
        data    = np.zeros((n, width, height, 3))
    else:
        data    = np.zeros((n, width, height, c_img)) # blank background for each image

    bboxes          = np.empty((n, 4))
    locations       = np.empty((n, 2))
    locations_norm  = np.empty((n, 2))
    nb_distractors  = np.empty((n, ))

    for k in range(n):

        # sample location
        x_pos   = np.random.randint(0,width - width_img)
        y_pos   = np.random.randint(0,height - height_img)

        # store locations
        # central location, format [-1,1]
        if norm:
            bboxes[k,:]         = (x_pos/float(width))*2.0 - 1.0, (y_pos/float(height))*2.0 - 1.0, width_img, height_img
            locations[k,:]      = x_pos+halfwidth_img, y_pos+halfheight_img
            locations[k,:]      = (locations[k,:]/float(width))*2.0 - 1.0
        else:
            bboxes[k,:]         = x_pos, y_pos, width_img, height_img
            locations[k,:]      = x_pos+halfwidth_img, y_pos+halfwidth_img

        # insert in blank image

        if color_noise or color_digits:
            if color_digits:
                colour = np.random.choice(np.arange(3))
                data[k, x_pos:x_pos+width_img, y_pos:y_pos+height_img, colour] += np.squeeze(batch[k])
            else:
                data[k, x_pos:x_pos+width_img, y_pos:y_pos+height_img, :] += np.tile(batch[k], (1,1,3))

        else:
            data[k, x_pos:x_pos+width_img, y_pos:y_pos+height_img, :] += batch[k]

        # add 8 x 8 sub-patches from random other digits
        n_distractors       = np.random.choice(np.arange(lim[0],lim[1]))
        nb_distractors[k]   = n_distractors

        for i in range(n_distractors):
            digit   = train_data[np.random.randint(0, train_data.shape[0]-1)]
            c1, c2  = np.random.randint(0, width_img - width_sub, size=2)
            i1, i2  = int(np.random.randint(0, width - width_sub, size=1)), int(np.random.randint(0, height - width_sub, size=1))

            if color_noise:
                colour = np.random.choice(np.arange(3))
                data[k, i1:i1+width_sub, i2:i2+height_sub, colour] += np.squeeze(digit[c1:c1+width_sub, c2:c2+height_sub, :])
            else:
                data[k, i1:i1+width_sub, i2:i2+height_sub, :] += digit[c1:c1+width_sub, c2:c2+height_sub, :]

    data = np.clip(data, 0., 1.)

    return data, locations, bboxes, nb_distractors

def plot_samples(X, locations, bboxes, nb_distractors, grid=(4,4), plot_bboxes=False, file_name='plot.pdf'):

        # params
        N, W, H, C = X.shape
        hw = W/2.

        assert grid[0]*grid[1] == X.shape[0]

        fig, axes = plt.subplots(grid[0],grid[1])

        for i, cur_ax in enumerate(axes.flat):

            # plot image
            if C == 1:
                cur_ax.matshow(np.squeeze(X[i]), cmap=plt.cm.gray)
            elif C == 3:
                cur_ax.imshow(np.squeeze(X[i]))
            else:
                raise Exception('C = {}'.format(C))

            # plot ground truth location
            if plot_bboxes:
                #cur_ax.scatter((locations[i,1]+1)*hw, (locations[i,0]+1)*hw, 30, marker='x', color=[1,1,1])

                # bounding box
                bbox = create_bbox([(bboxes[i,1]+1)*hw,(bboxes[i,0]+1)*hw,bboxes[i,2],bboxes[i,3]], color=[1,1,1])
                cur_ax.add_patch(bbox)

            cur_ax.set_axis_off()
            cur_ax.set_title('{} distractors'.format(nb_distractors[i]))

        plt.savefig('/home/janto/projects/ram/figures/tasks/{}'.format(file_name))
        plt.close()


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    import  matplotlib.pyplot as plt
    from    GlimpseNetwork import *
    from src.fig import create_bbox
    import cv2

    data = input_data.read_data_sets('MNIST_data', one_hot=True)
    N, K        = 10, 4
    W, H        = 500,500 # background size
    n_patches   = 4 # noise level for cluttered
    norm        = True
    lim         = (50,150)
    plot_bboxes = True

    mnist = data.train.images.reshape(-1,28,28,1)

    for i in range(N):

        # K samples
        sample_idx = np.random.permutation(mnist.shape[0])[:K]
        X          = mnist[sample_idx]

        if True:
            translated, locations, bboxes, nd = clutter_rnd(X,
                                                        mnist,
                                                        width=W, height=H,
                                                        lim=lim,
                                                        color_digits=True,
                                                        color_noise=True,
                                                        norm=norm)

        if False:
            translated, locations, bboxes = clutter_loc(X,
                                                        mnist, n_patches=n_patches,
                                                        width=W, height=H,
                                                        norm=norm)


        # plot
        plot_samples(translated, locations, bboxes, nd, grid=(1,4),
                     plot_bboxes=plot_bboxes,
                     file_name='m-cmnist_{}x{}_{}-{}_sample={}.png'.format(W,H,lim[0],lim[1],i))
