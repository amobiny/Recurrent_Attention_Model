"""
Defines all hyperparameters for DRAW model.
"""

class Config(object):

    # labels
    p_labels            = 0.5 # fraction of data with location ground truth
    gamma_start         = 1.0
    gamma_min           = 0.0

    # glimpse network
    convert_ratio       = 0.8
    original_size       = 28
    new_size            = 200
    num_channels        = 1  # remove later

    glimpse_size        = 12
    bandwidth           = glimpse_size**2

    n_patches           = 2 # # patches at each t
    num_glimpses        = 4 # samples before decision
    scale               = 2 # how much receptive field is scaled up for each glimpse

    sensor_size         = glimpse_size**2 * n_patches * num_channels

    minRadius           = 8
    hg_size = hl_size   = 128
    loc_dim             = 2
    g_size              = 256

    # logging

    # training
    batch_size          = 32
    eval_batch_size     = 50
    step                = 25000
    lr_start            = 1e-3
    lr_min              = 1e-5
    loc_std             = 0.11
    max_grad_norm       = 5.
    n_verbose           = 250

    # lstm
    cell_output_size    = 256
    cell_size           = 256
    cell_out_size       = cell_size

    # task
    num_classes         = 10
    n_distractors       = 4         # nr of digit distractors for cluttered task

    distractor_range    = (4,50)    # (lower,upper) limit of distractors for variable clutter tasks
    color_digits        = True
    color_noise         = True

    # monte carlo sampling
    M                   = 10

    # conv net
    conv_layers         = {0: [32, 7], 1: [32, 3], 2: [32,3]} # {0: [32, 5], 1: [32, 3]}
    coarse_size         = 80 #24
