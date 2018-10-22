""" Evaluates train model and produces plots

--load          checkpoint directory if you want to continue training a model
--task          task to train on {'mnist', 'translated' or 'cluttered'}
--num_glimpses  # glimpses (fixed)
--n_patches     # resolutions extracted for each glimpse

"""
import matplotlib as mpl
mpl.use('Agg')

import tensorflow as tf
import numpy as np
import os
import argparse
from   datetime import datetime
import pickle
from RAM        import RAM
from DRAM       import DRAM
from DRAM_loc   import DRAMl
from config     import Config
from src.utils  import evaluate_repeatedly

from tensorflow.examples.tutorials.mnist import input_data

# ----- parse command line -----
parser = argparse.ArgumentParser()
parser.add_argument('--task','-t', type=str, default='cluttered_var',
                    help='Task - ["org","translated","cluttered", "cluttered_var"].')

parser.add_argument('--model','-m', type=str, default='dram_loc',
                    help='Model - "RAM" or "DRAM".')
parser.add_argument('--load','-l', type=str, default=None,
                    help='Load model from directory.')
parser.add_argument('--num_glimpses','-n', type=int, default=8,
                    help='Number of glimpses to take')
parser.add_argument('--n_patches','-np', type=int, default=2,
                    help='Number of patches for each glimpse')
parser.add_argument('--use_context', default=False, action='store_true',
                    help='Use context network (True) or not (False)')
parser.add_argument('--convnet', default=False, action='store_true',
                    help='True: glimpse sensor is convnet, False: fully-connected')


parser.add_argument('--N','-N', type=int, default=10,
                    help='Number of plots')
parser.add_argument('--plot_dir','-od', type=str, default='plots',
                    help='Plot directory.')
parser.add_argument('--visualize','-v', default=False, action='store_true',
                    help='Create plots or not')
FLAGS, _ = parser.parse_known_args()


def evaluate(model='ram', width=60, n_distractors=4, N=10):
    """Tests performance of trained model on larger/noisier mnist images."""

    # data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    # set parameters
    #config.loc_std          = 1e-10
    config.num_glimpses     = FLAGS.num_glimpses
    config.n_patches        = FLAGS.n_patches
    config.use_context      = FLAGS.use_context
    config.convnet          = FLAGS.convnet
    config.sensor_size      = config.glimpse_size**2 * config.n_patches * config.num_channels
    config.N                = mnist.train.num_examples # number of training examples

    config.new_size         = width
    config.n_distractors    = n_distractors

    # init model
    print '\n-- Model: {} --'.format(model)
    print 'Setting samplding SD to {:.4e}'.format(config.loc_std)
    if model == 'ram':
        net = RAM(config)
    elif model == 'dram':
        net = DRAM(config)
    elif model == 'dram_loc':
        net = DRAMl(config)
    else:
        print 'Unknown model {}'.format(model)
        exit()


    net.load(FLAGS.load)  # restore
    net.count_params()

    #params = net.return_params(['context_network/conv0/w:0'])
    #net.plot_filters(params[0], fname=FLAGS.plot_dir + '.pdf')
    #exit()

    if FLAGS.visualize:

        # create plot for current parameters
        plot_dir = os.path.join(FLAGS.load, FLAGS.plot_dir)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        task                = {'variant': FLAGS.task, 'width': width, 'n_distractors': n_distractors}
        net.visualize(data=mnist,
                      task=task,
                      config=config,
                      N=N,
                      plot_dir=plot_dir)

    # evaluate
    #test, val = net.evaluate(data=mnist, task=FLAGS.task)

    return test, val


def evaluate_generalization(model='dram_loc', visualize=False, N=10):
    """Tests performance of trained model on larger/noisier mnist images."""

    # data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    # set parameters
    n_reps          = N
    widths          = [200]
    noise_levels    = [4]

    RESULTS         = {}

    for width in widths:
        for noise in noise_levels:

            # set parameters
            #config.loc_std          = 1e-10

            config.num_glimpses     = FLAGS.num_glimpses
            config.n_patches        = FLAGS.n_patches
            config.use_context      = FLAGS.use_context
            config.convnet          = FLAGS.convnet

            config.sensor_size      = config.glimpse_size**2 * config.n_patches * config.num_channels
            config.N                = mnist.train.num_examples # number of training examples

            config.new_size         = width
            config.n_distractors    = noise

            # init model
            print '\n-- Model: {} --'.format(model)
            print 'Setting samplding SD to {:.4e}'.format(config.loc_std)
            tf.reset_default_graph()
            if model == 'ram':
                net = RAM(config)
            elif model == 'dram':
                net = DRAM(config)
            elif model == 'dram_loc':
                net = DRAMl(config)
            else:
                print 'Unknown model {}'.format(model)
                exit()
            net.load(FLAGS.load)  # restore

            if FLAGS.visualize:
                n_reps =1

                # create plot for current parameters
                subfolder = os.path.join(FLAGS.load, FLAGS.plot_dir)
                if not os.path.exists(subfolder):
                    os.mkdir(subfolder)
                plot_dir  = os.path.join(subfolder, 'w={}_n_distractors={}'.format(width,noise))
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)

                task                = {'variant': FLAGS.task, 'width': width, 'n_distractors': noise}
                net.visualize(data=mnist,
                              task=task,
                              config=config,
                              plot_dir=plot_dir,
                              N=N)

            # evaluate (n_reps) times
            acc, _ = evaluate_repeatedly(ram=net, data=mnist, task=FLAGS.task, n_reps=n_reps)
            print acc

            # store results
            RESULTS[(width,noise)] = acc

    # save dictionary
    with open(os.path.join(FLAGS.load, 'glimpses{}_results.pickle'.format(FLAGS.num_glimpses )), 'wb') as handle:
        pickle.dump(RESULTS, handle, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate_numglimpses(model='dram_loc', visualize=False, N=10):
    """Tests performance of trained model on larger/noisier mnist images."""

    # data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    # set parameters
    n_glimpses      = [1,2,3,4,5,6,7,8]
    n_reps          = N
    width, noise    = 100, 4

    RESULTS         = {}

    for n in n_glimpses:

        # set parameters
        config.num_glimpses     = n
        config.n_patches        = FLAGS.n_patches
        config.use_context      = FLAGS.use_context
        config.convnet          = FLAGS.convnet

        config.sensor_size      = config.glimpse_size**2 * config.n_patches * config.num_channels
        config.N                = mnist.train.num_examples # number of training examples

        config.new_size         = width
        config.n_distractors    = noise

        # init model
        print '\n-- Model: {} --'.format(model)
        print 'Setting samplding SD to {:.4e}'.format(config.loc_std)
        tf.reset_default_graph()
        if model == 'ram':
            net = RAM(config)
        elif model == 'dram':
            net = DRAM(config)
        elif model == 'dram_loc':
            net = DRAMl(config)
        else:
            print 'Unknown model {}'.format(model)
            exit()
        net.load(FLAGS.load)  # restore

        if FLAGS.visualize:
            n_reps =1

            # create plot for current parameters
            subfolder = os.path.join(FLAGS.load, FLAGS.plot_dir)
            if not os.path.exists(subfolder):
                os.mkdir(subfolder)
            plot_dir  = os.path.join(subfolder, 'w={}_n_distractors={}'.format(width,noise))
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)

            task                = {'variant': FLAGS.task, 'width': width, 'n_distractors': noise}
            net.visualize(data=mnist,
                          task=task,
                          config=config,
                          plot_dir=plot_dir,
                          N=N)

        # evaluate (n_reps) times
        acc, _ = evaluate_repeatedly(ram=net, data=mnist, task=FLAGS.task, n_reps=n_reps)
        print acc

        # store results
        RESULTS[n] = acc

    # save dictionary
    with open(os.path.join(FLAGS.load, 'glimpses{}_results.pickle'.format(FLAGS.num_glimpses )), 'wb') as handle:
        pickle.dump(RESULTS, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    if FLAGS.model == 'ram':
        from config import Config
    elif FLAGS.model == 'dram':
        from config_dram import Config
    elif FLAGS.model == 'dram_loc':
        from config_dram import Config
    else:
        print 'Unknown model {}'.format(FLAGS.model)
        exit()

    # parameters
    config  = Config()
    n_steps = config.step

    # number of glimpses
    config.num_glimpses = FLAGS.num_glimpses
    config.n_patches    = FLAGS.n_patches
    config.N            = 55000 # number of training examples

    print '\n\nFlags: {}\n\n'.format(FLAGS)
    # ------------------------------

    #evaluate_generalization(model=FLAGS.model, N=2)

    # number of glimpses
    #evaluate_numglimpses(model=FLAGS.model, N=5)

    evaluate(model=FLAGS.model, width=config.new_size, n_distractors=config.n_distractors,
             N=FLAGS.N)
