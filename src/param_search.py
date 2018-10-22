"""
Hyperparameter grid-search.


--load          checkpoint directory if you want to continue training a model
--task          task to train on {'mnist', 'translated' or 'cluttered'}
--num_glimpses  # glimpses (fixed)
--n_patches     # resolutions extracted for each glimpse

"""

import  tensorflow as tf
import  numpy as np
import  argparse
from    datetime import datetime
import  pickle
from    RAM    import RAM
from    DRAM   import DRAM
from    config import Config
import  os

from tensorflow.examples.tutorials.mnist import input_data

# parameters
config          = Config()

def random_search(architecture='dram', task='translated', grid={}, k=5, config=None):
    """ Evaluates 'k' random hyper-parameter combinations and returns dict containing scores.

    Parameter names are (same as in config):
        * num_glimpses  (discrete)
        * glimpse_size  (discrete)
        * loc_std       (continuous)

    :param      architecture:   (str) 'ram' or 'dram'
    :param      grid:           (dict) {'parameter_name': np.range(lower,upper)}
    :param      k:              (int) # of combinations tested
    :return:    results         (dict) accuracy for each combination
    """

    print '\n\nModel {}\n\n'.format(architecture)

    # init
    results         = {}
    best_score      = 0 # higher is better
    best_params     = []

    # data
    mnist       = input_data.read_data_sets('MNIST_data', one_hot=False)
    config.N    = mnist.train.num_examples

    for i in range(k):

        # sample parameter combination
        config.num_glimpses         = np.random.choice(grid['num_glimpses'])
        config.loc_std              = np.round(np.random.choice(grid['loc_std']).astype(np.float32),2)
        config.glimpse_size         = np.random.choice(grid['glimpse_size'])

        # TODO: add other params here

        config.sensor_size         = config.glimpse_size**2 * config.n_patches
        config.bandwidth           = config.glimpse_size**2


        print '\n\n---- Combination #{} ----\n\n'.format(i)
        print 'Num glimpses:\t{}'.format(config.num_glimpses)
        print 'Glimpse size:\t{}x{}'.format(config.glimpse_size,config.glimpse_size)
        print 'Loc std:\t{}\n'.format(config.loc_std)

        cur_params = (config.num_glimpses, config.loc_std, config.glimpse_size)

        # log directory
        time_str = datetime.now().strftime('%H%M%S')

        logdir = "./parameter_search/model={}_{}{}x{}_n_glimpses={}_fovea={}x{}_std={:2.3f}_{}_lr={}-{}".format(
        architecture, task, config.new_size, config.new_size,config.num_glimpses, config.glimpse_size,config.glimpse_size,
        config.loc_std, time_str, config.lr_start, config.lr_min)

        # init model
        tf.reset_default_graph()

        if architecture == 'ram':
            model = RAM(config, logdir=logdir)
        elif architecture == 'dram':
            model = DRAM(config, logdir=logdir)
        else:
            print 'Unknown model {}'.format(architecture)
            exit()

        # train model
        model.train(mnist, task)

        # evaluate
        test, val = model.evaluate(data=mnist, task=task)

        print '\n\nResults:\nTest:\t{}\nVal:\t{}'.format(test,val)

        # store
        results[cur_params] = [test,val]

        # update best score
        if test > best_score:
            print 'Best params: {}\tScore: {}'.format(cur_params, test)
            results['best_params']  = cur_params
            best_score              = test

    return results



if __name__ == '__main__':

    # parameter grid
    grid = {
        'num_glimpses':     [6],
        'loc_std':          np.arange(0.03, 0.09, 0.005),
        'glimpse_size':     [6,8]
    }

    # search
    results = random_search(architecture='ram', task='translated', grid=grid, k=8, config=config)

    print '\n\n{}\n\n'.format(results)

    # save results
    path = os.path.join('./parameter_search/results.p')

    pickle.dump(results, open(path, 'wb'))
    print 'Saved results to <<{}>>'.format(path)
