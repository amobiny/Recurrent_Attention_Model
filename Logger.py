""" Implements Logger class for tensorflow experiments. """

import tensorflow as tf
import numpy as np
import os


class Logger(object):
    """
    Writes summaries and saves checkpoints for experiments.
  """

    def __init__(self, log_dir='./log', sess=None, summary_ops={}, var_list=[],
                 global_step=None, eval_ops={}, n_verbose=1000):
        """

    :param log_dir:       log path
    :param sess:          Session
    :param summary_ops:   (dict) keys: name, values: summary operator
    :param eval_ops:      (dict) keys: name, values: operator (same keys as summary_ops)
    :param n_verbose:     how often to print statistics
    """

        self.session = sess

        # folders
        self.log_dir = log_dir
        self.checkpoint_path, self.summary_path = create_directories(log_dir)

        # file writers
        self.writers = {
            'train': tf.summary.FileWriter(os.path.join(self.summary_path, 'train'), self.session.graph),
            'test': tf.summary.FileWriter(os.path.join(self.summary_path, 'test')),
            'val': tf.summary.FileWriter(os.path.join(self.summary_path, 'val'))
        }

        # saver
        self.global_step = global_step
        if var_list == []:
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        else:
            self.saver = tf.train.Saver(var_list, keep_checkpoint_every_n_hours=1)

        # summaries
        self.summary_ops = summary_ops
        self.eval_ops = eval_ops
        self.merged_op = tf.summary.merge_all()

        # step counter
        self.step = 0
        self.n_verbose = n_verbose

    def log(self, writer_id, feed_dict):
        """ Logs performance using either 'train', 'test' or 'val' writer"""

        summaries = self.session.run(self.merged_op, feed_dict=feed_dict)
        self.writers[writer_id].add_summary(summaries, self.step)

        if self.n_verbose and self.step % self.n_verbose == 0:
            print '\n------ Step {} ------'.format(self.step)

            for key in self.eval_ops.keys():
                val = self.session.run(self.eval_ops[key], feed_dict)
                print '{}={:3.4f}\t'.format(key, val),

    def save(self):
        """Saves model checkpoint."""
        self.saver.save(self.session, os.path.join(self.checkpoint_path, 'checkpoint'), global_step=self.global_step)

    def restore(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt
            self.saver.restore(self.session, ckpt.model_checkpoint_path)


def create_directories(log_dir):
    """ Creates logging and checkpoint directories. """

    # create directory and subfolders
    checkpoint_path = os.path.join(log_dir, 'checkpoints')
    summary_path = os.path.join(log_dir, 'summaries')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        os.mkdir(checkpoint_path)
        os.mkdir(summary_path)

        print '\n\nLogging to <<{}>>.\n\n'.format(log_dir)

    return checkpoint_path, summary_path
