# Plotting utilities
import matplotlib

matplotlib.use('Agg')

# base
import os
import sys
import glob
import time
import tensorflow as tf

# external
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.patches as patches

sns.set_style("white")


def create_bbox(bb, color='red', linewidth=1.5, alpha=1.0):
    """Returns bounding box rectangle to plot.to

    Args:
        bb      -- (list/array) (1 x 4) containing (x_lowerleft, y_lowerleft, x_width, y_width)
        color   -- (str)
    Output:
        bbox    -- (matplotlib Rectangle) Rectangle object. Add to plot:
                        >> plt.imshow(X)                # base image
                        >> bbox = create_bbox([1,2,30,30])
                        >> plt.gca().add_patch(bbox)    # add bounding box
    """
    return mpatches.Rectangle((bb[0], bb[1]), bb[2], bb[3],
                              edgecolor=color, fill=False, linewidth=linewidth, alpha=alpha)


def plot_glimpses(config=None, glimpse_images=[], pred_labels=[], probs=[], sampled_loc=[],
                  X=[], labels=[], file_name=[], fontsize=5, keep=True):
    """ For each glimpse plots one .png with:
            * glimpse location on on original image (panel 1)
            * extracted patches at different resolutions (panel 2:end)

    Args:
    sess                   -- tf.Session
    sampled_loc_arr        -- (list) rnn decoder ops
    sampled_loc_arr        -- (4D tensor) N x W x H x C images
    grid                   -- (tuple) grid dimensions (*optional), o.w. automatically calculated
    file_name              -- (str) if specifed saves plot here (*optional)
    fontsize               -- (int) title fontsize
    """

    # dimensions
    N, H, W, C = X.shape
    n_glimpses = config.num_glimpses
    glimpse_w_half = config.glimpse_size / 2.0

    assert N < 80, 'Only plots < 80 plots.'
    assert X.ndim == 4, '(X) must have 4 dimensions (N x H x W x C).'

    for i in range(N):

        # locations of current example
        cur_loc = sampled_loc[i, :, :]

        # normalize
        cur_loc_norm = ((cur_loc + 1) * config.new_size / 2.0).astype(int)

        for glimpseI in range(n_glimpses):

            # init plot
            if probs == []:
                fig, cur_ax = plt.subplots(1, config.n_patches + 1)
            else:
                fig, cur_ax = plt.subplots(1, config.n_patches + 1 + 1)

            # 1. plot glimpses
            if C == 1:  # greyscale images
                cur_ax[0].matshow(np.squeeze(X[i, :, :]), cmap=plt.cm.gray)

            else:  # RGB images
                cur_ax[0].imshow(X[i])

            cur_ax[0].set_axis_off()

            # 2. plot different resolution patches
            # glimpses = glimpse_net.get_glimpse(loc_ph)
            glimpses = glimpse_images[i, glimpseI]

            for k in range(config.n_patches):

                if C == 1:
                    cur_ax[1 + k].matshow(np.squeeze(glimpses[k]), cmap=plt.cm.gray)
                else:
                    cur_ax[1 + k].imshow(np.squeeze(glimpses[k]))

                cur_ax[1 + k].set_axis_off()

            # 3. plot probabilities
            if probs != []:
                if glimpseI < config.num_glimpses:
                    cur_ax[-1].bar(range(10), probs[i, glimpseI, :], color='b')
                    cur_ax[-1].set_aspect(3)
                    cur_ax[-1].set_ylim((0, 1))
                    cur_ax[-1].set_xticks(range(10))
                    cur_ax[-1].set_xlabel('Digit')
                    cur_ax[-1].set_title('Probabilities')

                    # color code correct/wrong for each t
                    if np.argmax(probs[i, glimpseI, :]) == labels[i]:
                        color = 'limegreen'
                    else:
                        color = 'r'
            else:  # use predicted labels
                if pred_labels[i] == labels[i]:
                    color = 'limegreen'
                else:
                    color = 'r'
                # color = [1.,1.,1.]

            if glimpseI == n_glimpses - 1:
                linewidth = 3
            else:
                linewidth = 1.5

            # add glimpse boxes
            if keep:
                alpha_step = 1.0 / n_glimpses
                alpha = 0
                for h in range(glimpseI + 1):
                    alpha += alpha_step
                    add_glimpses(axis=cur_ax[0], loc=cur_loc_norm[h, :], w=config.glimpse_size,
                                 n_patches=config.n_patches,
                                 color=color, linewidth=linewidth, alpha=alpha)

            else:
                add_glimpses(axis=cur_ax[0], loc=cur_loc_norm[glimpseI, :], w=config.glimpse_size,
                             n_patches=config.n_patches, color=color, linewidth=linewidth)

            png_name = file_name + '_n={}_glimpse={}.png'.format(i, glimpseI)
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            plt.savefig(png_name, bbox_inches='tight')
            plt.close()

    return


def plot_trajectories(config=None, locations=[],
                      X=[], labels=[], pred_labels=[],
                      grid=[], file_name=[], bboxes=[], fontsize=5, alpha=0.5):
    """ Plots glimpse trajectories over time and saves plots as png files.

    Args:
        config                 -- params
        locations              -- (np.array)  N x S x 2 (S:= # glimpses + 1) (y,x) locations in [-1,1]
        X                      -- (np.array)  N x W x H x C images
        labels                 -- (np.array)  N x 1 ground truth
        pred_labels            -- (np.array)  N x 1 predicted labels

        grid                   -- (tuple) grid dimensions (*optional), o.w. automatically calculated
        file_name              -- (str) if specifed saves plot here (*optional)
        fontsize               -- (int) title fontsize
    """
    # dimensions
    N, H, W, C = X.shape
    n_glimpses = len(locations)

    hw = W / 2.

    assert N < 80, 'Only plots < 80 plots.'
    assert X.ndim == 4, '(X) must have 4 dimensions (N x H x W x C).'

    if grid != []:
        assert grid[0] * grid[1] == N, 'Grid must have as many subplots as passed images. {} != {}.'.format(
            grid[0] * grid[1], N)

    if grid == []:
        # determine number of subplots (square)
        n_rows = int(np.ceil(np.sqrt(N)))
        n_cols = n_rows
    else:
        n_rows, n_cols = grid

    # init plot
    fig, ax = plt.subplots(n_rows, n_cols)

    # convert locations to (x,y) pixel coordinates
    # exclude last coordinate
    pixel_locations = ((locations[:, :-1, :] + 1) * config.new_size / 2.0).astype(int)

    for i, cur_ax in enumerate(ax.flat):

        if i < N:

            # plot base image
            if C == 1:  # grayscale images
                cur_ax.matshow(np.squeeze(X[i, :, :]), cmap=plt.cm.gray)

            else:  # RGB images
                cur_ax.imshow(X[i])

            cur_ax.set_axis_off()
            if len(labels) != 0:  # if non-empty
                cur_ax.set_title('Truth {} | Pred {}'.format(labels[i], pred_labels[i]), fontsize=fontsize)

            # add glimpse trajectory
            cur_locations = np.squeeze(pixel_locations[i, :, :])  # all locations for current image

            # color code correct/wrong
            # print 'Label {} - Pred {}'.format(y_hat[i], labels[i])
            if pred_labels[i] == labels[i]:
                color = 'limegreen'
            else:
                color = 'r'

            cur_ax.plot(cur_locations[:, 1], cur_locations[:, 0], '-', color=color, linewidth=1.5, alpha=alpha)
            cur_ax.scatter(cur_locations[0, 1], cur_locations[0, 0], 15, facecolors='none', linewidth=1.5, color=color,
                           alpha=alpha)
            cur_ax.plot(cur_locations[-1, 1], cur_locations[-1, 0], 'o', color=color, markersize=5, alpha=alpha)
            if bboxes != []:
                bbox = create_bbox([(bboxes[i, 1] + 1) * hw, (bboxes[i, 0] + 1) * hw, bboxes[i, 2], bboxes[i, 3]],
                                   color=[1, 1, 1], alpha=0.3, linewidth=0.7)
                cur_ax.add_patch(bbox)
        else:
            fig.delaxes(cur_ax)

    plt.tight_layout(pad=0.2)

    if file_name == []:
        plt.show()
    else:
        # print 'Saved plot to {}'.format(file_name)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    return


def add_glimpses(axis, loc=[], n_patches=3, w=8, color='r', linewidth=1.5, alpha=1.0):
    """ Adds three glimpse boxes to current axis.
    Args:
        axis        current axis
        loc         (1x2 np.array) glimpse center
        n_patches   number of glimpses
        w           width of glimpse
    """
    for k in range(n_patches):  # range(n_patches):

        mult = 2 ** k
        hw = mult * w / 2.0

        box = [int(loc[1] - hw), int(loc[0] - hw), mult * w, mult * w]
        axis.add_patch(create_bbox(box, color=color, linewidth=linewidth, alpha=alpha))  # ground truth bounding boxes

    return


def norm2ind(norm_ind, width):
    """Converts from (-1,1) to pixel indices.

    :param norm_ind     2D array containing (y,x) coordinates in (-1,1)
    :param width        image width
    """
    return width * ((norm_ind + 1) / 2.0).astype(int)
