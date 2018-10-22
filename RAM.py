"""
Implements Recurrent Attention Model (RAM) based on [1]

[1] Mnih et al. 2014. Recurrent Attention Model.
"""

import tensorflow as tf
import numpy as np
import os

from GlimpseNetwork import GlimpseNetwork, LocNet
from Logger import Logger
from data_generator import *
from src.utils import *
from src.fig import plot_glimpses, plot_trajectories

# tensorflow version switch
rnn_cell = tf.contrib.rnn
seq2seq = tf.contrib.legacy_seq2seq


class RAM(object):
    """Implements RAM model."""

    def __init__(self, config, logdir='.'):

        # parameters
        self.config = config
        self.logdir = logdir

        # input placeholders
        self.images_ph = tf.placeholder(tf.float32,
                                        [None, self.config.new_size, self.config.new_size, self.config.num_channels])
        self.labels_ph = tf.placeholder(tf.int64, [None])
        self.N = tf.shape(self.images_ph)[0]  # number of examples

        # glimpse network
        with tf.variable_scope('glimpse_net'):
            self.gl = GlimpseNetwork(self.config, self.images_ph)
        with tf.variable_scope('loc_net'):
            self.loc_net = LocNet(self.config)

        # initial glimpse
        # self.init_loc           = tf.random_uniform((self.N, 2), minval=-0.5, maxval=0.5)
        self.init_loc = tf.zeros(shape=[self.N, 2], dtype=tf.float32, )

        self.init_glimpse = self.gl(self.init_loc)
        self.inputs = [self.init_glimpse]
        self.inputs.extend([0] * (self.config.num_glimpses - 1))

        # ------- Core: recurrent network -------
        self.loc_mean_arr = []
        self.sampled_loc_arr = []
        self.glimpses = []
        self.glimpses.append(self.gl.glimpse_img)

        def get_next_input(output, i):
            """Samples next glimpse location."""
            loc, loc_mean = self.loc_net(output)  # takes hidden RNN state and produces next location
            gl_next = self.gl(loc)

            # for visualization
            self.loc_mean_arr.append(loc_mean)
            self.sampled_loc_arr.append(loc)
            self.glimpses.append(self.gl.glimpse_img)

            return gl_next

        self.lstm_cell = rnn_cell.LSTMCell(self.config.cell_size, state_is_tuple=True, activation=tf.nn.relu)
        self.init_state = self.lstm_cell.zero_state(self.N, tf.float32)

        # output: list of num_glimpses + 1
        self.outputs, _ = seq2seq.rnn_decoder(
            self.inputs, self.init_state, self.lstm_cell, loop_function=get_next_input)
        get_next_input(self.outputs[-1], 0)
        # ---------------------------------------

        # time independent baselines
        with tf.variable_scope('baseline'):
            w_baseline = weight_variable((self.config.cell_output_size, 1))
            b_baseline = bias_variable((1,))
        baselines = []

        for t, output in enumerate(self.outputs):  # ignore initial state
            baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
            baseline_t = tf.squeeze(baseline_t)
            baselines.append(baseline_t)

        # outputs for each glimpse (t)
        baselines = tf.stack(baselines)  # [timesteps, batch_size]
        self.baselines = tf.transpose(baselines)  # [batch_size, timesteps]

        # Take the last step only.
        self.output = self.outputs[-1]  # [batch size x cell_output_size]

        # ---- for visualizations ----
        self.sampled_locations = tf.concat(self.sampled_loc_arr, axis=0)
        self.mean_locations = tf.concat(self.loc_mean_arr, axis=0)

        # self.sampled_locs = tf.reshape(self.sampled_locs, (self.batch_size, self.glimpses, 2))
        self.sampled_locations = tf.reshape(self.sampled_locations, (self.config.num_glimpses, self.N, 2))
        self.sampled_locations = tf.transpose(self.sampled_locations, [1, 0, 2])

        self.mean_locations = tf.reshape(self.mean_locations, (self.config.num_glimpses, self.N, 2))
        self.mean_locations = tf.transpose(self.mean_locations, [1, 0, 2])

        prefix = tf.expand_dims(self.init_loc, 1)
        self.sampled_locations = tf.concat([prefix, self.sampled_locations], axis=1)
        self.mean_locations = tf.concat([prefix, self.mean_locations], axis=1)

        self.glimpses = tf.stack(self.glimpses, axis=1)
        # -----------------------------

        # classification network
        with tf.variable_scope('classification'):
            w_logit = weight_variable((self.config.cell_output_size, self.config.num_classes))
            b_logit = bias_variable((self.config.num_classes,))

            self.logits = tf.nn.xw_plus_b(self.output, w_logit, b_logit)
            self.softmax = tf.nn.softmax(self.logits)  # [batch_size x n_classes]

            # class probabilities for each glimpse
            self.class_prob_arr = []

            for op in self.outputs:
                self.glimpse_logit = tf.stop_gradient(tf.nn.xw_plus_b(op, w_logit, b_logit))
                self.class_prob_arr.append(tf.nn.softmax(self.glimpse_logit))

            self.class_prob_arr = tf.stack(self.class_prob_arr, axis=1)
            # [batch_size x num_glimpses x n_classes]
        # Losses/reward

        # cross-entropy
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_ph)
        self.xent = tf.reduce_mean(xent)
        self.pred_labels = tf.argmax(self.logits, 1)

        # REINFORCE: 0/1 reward
        self.reward = tf.cast(tf.equal(self.pred_labels, self.labels_ph), tf.float32)

        self.rewards = tf.expand_dims(self.reward, 1)  # [batch_sz, 1]
        self.rewards = tf.tile(self.rewards, (1, self.config.num_glimpses))  # [batch_sz, timesteps]
        self.logll = loglikelihood(self.loc_mean_arr, self.sampled_loc_arr, self.config.loc_std)
        self.advs = self.rewards - tf.stop_gradient(self.baselines)
        self.logllratio = tf.reduce_mean(self.logll * self.advs)
        self.reward = tf.reduce_mean(self.reward)

        self.baselines_mse = tf.reduce_mean(tf.square((self.rewards - self.baselines)))  # original
        self.var_list = tf.trainable_variables()

        # hybrid loss
        self.loss = -self.logllratio + self.xent + self.baselines_mse  # `-` to minimize
        self.grads = tf.gradients(self.loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(self.grads, self.config.max_grad_norm)

        # set up optimization
        self.setup_optimization()

        # session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def setup_optimization(self, training_steps_per_epoch=None):
        """Set up optimzation operators."""

        # learning rate
        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        self.training_steps_per_epoch = self.config.N // self.config.batch_size
        # print 'Training steps / epoch {}'.format(self.training_steps_per_epoch)

        self.starter_learning_rate = self.config.lr_start
        # decay per training epoch
        self.learning_rate = tf.train.exponential_decay(
            self.starter_learning_rate,
            self.global_step,
            self.training_steps_per_epoch,
            0.97,
            staircase=True)
        self.learning_rate = tf.maximum(self.learning_rate, self.config.lr_min)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.var_list),
                                                       global_step=self.global_step)

    def setup_logger(self):
        """Creates log directory and initializes logger."""
        loc_net1 = [v for v in tf.global_variables() if v.name == "loc_net/Variable:0"]

        self.summary_ops = {
            'reward': tf.summary.scalar('reward', self.reward),
            'hybrid_loss': tf.summary.scalar('hybrid_loss', self.loss),
            'cross_entropy': tf.summary.scalar('cross_entropy', self.xent),
            'baseline_mse': tf.summary.scalar('baseline_mse', self.baselines_mse),
            'logllratio': tf.summary.scalar('logllratio', self.logllratio),
            'loc_net1': tf.summary.histogram('loc_net1', loc_net1),
            'glimpses': tf.summary.image('glimpses',
                                         tf.reshape(self.glimpses,
                                                    [-1, self.config.glimpse_size, self.config.glimpse_size,
                                                     self.config.num_channels]),
                                         max_outputs=20)
        }
        self.eval_ops = {
            'reward': self.reward,
            'hybrid_loss': self.loss,
            'cross_entropy': self.xent,
            'baseline_mse': self.baselines_mse,
            'logllratio': self.logllratio,
            'lr': self.learning_rate
        }
        self.logger = Logger(self.logdir, sess=self.session, summary_ops=self.summary_ops,
                             global_step=self.global_step, eval_ops=self.eval_ops,
                             n_verbose=self.config.n_verbose, var_list=self.var_list)

    def train(self, data=[], task='mnist'):
        """Trains RAM model and logs statistics.

        Args:
            data    -- data set object (.train, .test, .validation), cf mnist
            task    -- str ['mnist','translated','cluttered']
            data    -- data set object (.train, .test, .validation), cf mnist
        """
        # verbose
        print '\n\n\n------------ Starting training ------------  \nTask: {} -- {}x{} with {} distractors\n' \
              'Model: {} patches, {} glimpses, glimpse size {}x{}\n\n\n'.format(
            task, self.config.new_size, self.config.new_size, self.config.n_distractors,
            self.config.n_patches, self.config.num_glimpses, self.config.glimpse_size, self.config.glimpse_size
        )

        self.task = task
        self.setup_logger()  # add logger

        for i in xrange(self.config.step):

            images, labels = data.train.next_batch(self.config.batch_size)
            images = images.reshape((-1, self.config.original_size, self.config.original_size, 1))

            # choose task
            if self.task == 'translated':
                images = translate(images, width=self.config.new_size, height=self.config.new_size)
            elif self.task == 'cluttered':
                images = clutter(images, train_data=data.train.images.reshape(
                    (-1, self.config.original_size, self.config.original_size, 1)),
                                 width=self.config.new_size, height=self.config.new_size,
                                 n_patches=self.config.n_distractors)

            # duplicate M times, see Eqn (2)
            images = np.tile(images, [self.config.M, 1, 1, 1])
            labels = np.tile(labels, [self.config.M])
            self.loc_net.sampling = True

            # training step
            feed_dict = {self.images_ph: images, self.labels_ph: labels}
            self.session.run(self.train_op, feed_dict=feed_dict)

            # log
            self.logger.step = i
            self.logger.log('train', feed_dict=feed_dict)

            # evaluation on test/validation
            # if i and i % (2 * self.training_steps_per_epoch) == 0:
            if i and i % 1000 == 0:
                # save model
                self.logger.save()
                print '\n==== Evaluation: (step {}) ===='.format(i)
                self.evaluate(data, task=self.task)

    def evaluate(self, data=[], task='mnist'):
        """Returns accuracy of current model.

        Returns:
            test_accuracy, validation_accuracy

        """
        return evaluate(self.session, self.images_ph, self.labels_ph, self.softmax,
                        data, self.config, task)

    def load(self, checkpoint_dir):
        """Restores model from <<checkpoint_dir>>. Assumes sub-folder 'checkpoints' in directory."""

        folder = os.path.join(checkpoint_dir, 'checkpoints/')
        print '\nLoading model from <<{}>>.\n'.format(folder)

        self.saver = tf.train.Saver(self.var_list)

        ckpt = tf.train.get_checkpoint_state(folder)

        if ckpt and ckpt.model_checkpoint_path:
            print ckpt
            self.saver.restore(self.session, ckpt.model_checkpoint_path)

    def visualize(self, config=[], data=[], task={'variant': 'mnist', 'width': 60, 'n_distractors': 4},
                  plot_dir='.', N=10, seed=None):
        """Given a saved model visualizes inference.

        Args:
                config          params
                data            data object (cf mnist)
                task            (dict) parameters for task to evaluate on

                N               (int) number of plots
                seed            (int) random if 'None', seed='seed' o.w.
        """
        print '\n\nGenerating visualizations ....',

        np.random.seed(seed)

        # evaluation task
        self.loc_net.sampling = False

        # config.new_size         = task['width']
        # config.n_distractors    = task['n_distractors']

        # data
        X = data.test.images.reshape((-1, 28, 28, 1))
        labels = data.test.labels

        # test model
        # if task['variant'] == 'translated':
        #     X = translate(X, width=task['width'], height=task['width'])
        # elif task['variant'] == 'cluttered':
        #     X = clutter(X, X, width=task['width'], height=task['width'], n_patches=task['n_distractors'])
        # else:
        #     print 'Using original MNIST data.'

        # sample random subset of data
        idx = np.random.permutation(X.shape[0])[:N]

        X, Y = X[idx], labels[idx]

        # data for plotting
        feed_dict = {self.images_ph: X, self.labels_ph: Y}
        fetches = [self.glimpses, self.sampled_locations, self.mean_locations, self.pred_labels, self.class_prob_arr]

        results = self.session.run(fetches, feed_dict=feed_dict)
        glimpse_images, sampled_locations, mean_locations, pred_labels, probs = results

        # glimpses
        plot_glimpses(config=self.config, glimpse_images=glimpse_images, pred_labels=pred_labels, probs=[],
                      sampled_loc=mean_locations,
                      X=X, labels=Y, file_name=os.path.join(plot_dir, 'glimpses_mean'))

        plot_trajectories(config=self.config, locations=mean_locations,
                          X=X, labels=Y, pred_labels=pred_labels, file_name=os.path.join(plot_dir, 'trajectories'))

    def count_params(self):
        """Returns number of trainable parameters."""
        return count_parameters(self.session)
