"""
Modification of MultiRNNCell.

Original:       'MultiRNNCell' returns ('outputs', 'states') where 'outputs' is a list of outputs
                from the final layer (length 'n_steps')
Modification:   'JantosMultiRNNCell' returns ('outputs', 'states') where 'outputs' is a
                list of tuples of outputs from each layer.
                Each tuple contains all outputs at one time step, where entry i is the
                output from layer i.

                I only modified 'output_size' and the 'call' method
"""
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import *

class MyMultiRNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells, state_is_tuple=True):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    super(MyMultiRNNCell, self).__init__()
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    if not nest.is_sequence(cells):
      raise TypeError(
          "cells must be a list or tuple, but saw: %s." % cells)

    self._cells = cells
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    #return self._cells[-1].output_size
    return tuple(cell.output_size for cell in self._cells) # [added]

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._state_is_tuple:
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
      else:
        # We know here that state_size of each cell is not a tuple and
        # presumably does not contain TensorArrays or anything else fancy
        return super(MyMultiRNNCell, self).zero_state(batch_size, dtype)

  def __call__(self, inputs, state):
    """Run this multi-layer cell on inputs, starting from state."""
    cur_state_pos   = 0
    cur_inp         = inputs
    new_states      = []
    int_outputs     = [] # intermediate outputs, length 'len(self._cells)' # [added]

    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        if self._state_is_tuple:
          if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        cur_inp, new_state = cell(cur_inp, cur_state)
        new_states.append(new_state)
        int_outputs.append(cur_inp) # [added]

    new_states  = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))
    int_outputs = (tuple(int_outputs) if self._state_is_tuple else
                  array_ops.concat(int_outputs, 1))

    #print '[JantoMultiRNNCell], new_outputs.ndim {}'.format(int_outputs)
    #print '\tint_outputs type {}'.format(type(int_outputs))
    return int_outputs, new_states # [added]
