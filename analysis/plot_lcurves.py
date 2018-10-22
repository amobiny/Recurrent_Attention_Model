"""
Given a folder with .csv files containing tensorboard data
plots curves in one plot.
"""
import matplotlib
matplotlib.use('Agg')

# base
import  os
import  re
import  argparse
import  sys
import  glob
import  numpy               as np
import  pandas              as pd
import  matplotlib.pyplot   as plt
from matplotlib import cm
import  seaborn             as sns

sns.set_style("white")

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def smooth_conv(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


if __name__ == '__main__':

    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir','-d', type=str, default=None,
                        help='Data directory of .csv files.')
    parser.add_argument('--plot_name','-p', type=str, default='learning_curves.pdf',
                        help='Name of resulting plot')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--threshold','-th', type=float, default=0.9,
                        help='Threshold to count towards.')
    parser.add_argument('--smooth', default=False, action='store_true',
                        help='True: smooth, False: do not.')
    parser.add_argument('--smooth_method', type=str, default='flat',
                        help='["flat", "hanning", "hamming", "bartlett", "blackman"]')
    FLAGS, _ = parser.parse_known_args()
    # ------------------------------

    # plot parameters
    SETTINGS = {
        'linewidth': 1.5,
        'color':     'r',
        'dram_color': 'b', # color for 0% labeled data
        'alpha':     0
    }

    # parameters
    threshold   = FLAGS.threshold
    T           = -1  # how many steps to show
    n_points    = 100

    # for conv smoothing
    window_len  = 30
    #offset      = 15 # exclude first points to remove artifacts

    # read in data
    f = glob.glob(os.path.join(FLAGS.dir,'*accuracy.csv'))

    print '\n{} files found.\n'.format(len(f))

    # init axes
    fig, cur_ax         = plt.subplots(1,1)
    alpha, alpha_step   = 0, 1.0/len(f) # how fast to increase transparency

    # get line colors
    cm_subsection = np.linspace(0, 1, len(f))
    colors = [ cm.Reds(x) for x in cm_subsection ]

    # threshold
    cur_ax.axhline(threshold*100, linewidth=SETTINGS['linewidth'], color=[0.7,0.7,0.7])

    for i, file in enumerate(sorted(f)):

        # verbose
        file_name = os.path.split(file)[-1]

        # get % labels
        start       = file_name.find('labels=') + 7
        end         = file_name.find('_accuracy')
        file_name   =  file_name[start:end]
        print file_name

        # increase color intensity
        #alpha += alpha_step

        # data frame: columns ['wall time', 'step', 'value']
        data = pd.read_csv(file)

        # smoothed y data
        X           = list(data['step'][:T] * FLAGS.batch_size)
        walltime    = list(data['wall_time'][:T])
        start_time  = walltime[0]

        if i == 0:
            print('-- Plotting the first {} updates'.format(X[-1]))


        if FLAGS.smooth:
            Y = smooth(data['value'][:T], n_points)
            #Y = smooth_conv(data['Value'], window_len=window_len, window=FLAGS.smooth_method)[:T]
        else:
            Y = data['value'][:T]

        # compute # steps to treshold
        n_steps = np.where(Y > threshold)[0]
        n_steps = n_steps[0] if n_steps != [] else -1

        # transform Y to %
        Y *= 100

        # add to plot
        try:
            cur_ax.axvline(X[n_steps], linewidth=SETTINGS['linewidth']/2, linestyle='--',
                           color=colors[i])
            cur_ax.scatter(X[n_steps],Y[n_steps],20,color=colors[i])
        except:
            pass

        # cut off last steps
        X, Y = X[:-100], Y[:-100]

        if file_name == '0.00' or file_name == '0.0':
          cur_ax.plot(X, Y,
                linewidth=SETTINGS['linewidth'],
                color=SETTINGS['dram_color'],
                alpha=1.0,
                label='DRAM')

          base_speed = n_steps
          base_time  = walltime[n_steps]

          cur_ax.axvline(X[n_steps], linewidth=SETTINGS['linewidth']/2, linestyle='--',
                         color=SETTINGS['dram_color'],alpha=1.0)
          cur_ax.scatter(X[n_steps],Y[n_steps],20,
                         color=SETTINGS['dram_color'],
                         alpha=1.0
                         )
        else:
            cur_ax.plot(X, Y,
                        linewidth=SETTINGS['linewidth'],
                        #color=SETTINGS['color'],
                        #alpha=alpha,
                        color=colors[i],
                        label='L-DRAM, ' + str(float(file_name)*100) + '%')

        # print number of steps
        t = abs(start_time - walltime[n_steps])
        print('{:30s}\t{:6d} examples ({:6d})\t\t{:.2f}x speed-up\t{}'.format(
            file_name, X[n_steps], n_steps, base_speed / float(n_steps), t /60. ))



    # save plot
    cur_ax.set_ylim([0,100])
    cur_ax.set_xlabel('Nr. examples')
    cur_ax.set_ylabel('Training accuracy')

    plt.legend()
    #plt.grid('on')
    plt.savefig('th={}'.format(threshold) + FLAGS.plot_name, bbox_inches='tight')
    plt.close()



# """
# Given a folder with .csv files containing tensorboard data
# plots curves in one plot.
# """
# import matplotlib
# matplotlib.use('Agg')
#
# # base
# import  os
# import  re
# import  argparse
# import  sys
# import  glob
# import  numpy               as np
# import  pandas              as pd
# import  matplotlib.pyplot   as plt
# import  seaborn             as sns
#
# sns.set_style("white")
#
# def smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
#
# def smooth_conv(x,window_len=11,window='hanning'):
#     """smooth the data using a window with requested size.
#
#     This method is based on the convolution of a scaled window with the signal.
#     The signal is prepared by introducing reflected copies of the signal
#     (with the window size) in both ends so that transient parts are minimized
#     in the begining and end part of the output signal.
#
#     input:
#         x: the input signal
#         window_len: the dimension of the smoothing window; should be an odd integer
#         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#             flat window will produce a moving average smoothing.
#
#     output:
#         the smoothed signal
#
#     example:
#
#     t=linspace(-2,2,0.1)
#     x=sin(t)+randn(len(t))*0.1
#     y=smooth(x)
#
#     see also:
#
#     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
#     scipy.signal.lfilter
#
#     TODO: the window parameter could be the window itself if an array instead of a string
#     NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#     """
#
#     if x.ndim != 1:
#         raise ValueError, "smooth only accepts 1 dimension arrays."
#
#     if x.size < window_len:
#         raise ValueError, "Input vector needs to be bigger than window size."
#
#
#     if window_len<3:
#         return x
#
#
#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
#
#
#     s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#     #print(len(s))
#     if window == 'flat': #moving average
#         w=np.ones(window_len,'d')
#     else:
#         w=eval('np.'+window+'(window_len)')
#
#     y=np.convolve(w/w.sum(),s,mode='valid')
#     return y
#
#
# if __name__ == '__main__':
#
#     # ----- parse command line -----
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dir','-d', type=str, default=None,
#                         help='Data directory of .csv files.')
#     parser.add_argument('--plot_name','-p', type=str, default='learning_curves.pdf',
#                         help='Name of resulting plot')
#     parser.add_argument('--threshold','-th', type=float, default=0.9,
#                         help='Threshold to count towards.')
#     parser.add_argument('--smooth', default=False, action='store_true',
#                         help='True: smooth, False: do not.')
#     parser.add_argument('--smooth_method', type=str, default='flat',
#                         help='["flat", "hanning", "hamming", "bartlett", "blackman"]')
#     FLAGS, _ = parser.parse_known_args()
#     # ------------------------------
#
#     # plot parameters
#     SETTINGS = {
#         'linewidth': 1.5,
#         'color':     'r',
#         'dram_color': 'b', # color for 0% labeled data
#         'alpha':     0
#     }
#
#     # parameters
#     threshold   = FLAGS.threshold
#     T           = 300  # how many steps to show
#     n_points    = 500
#
#     # for conv smoothing
#     window_len  = 30
#     offset      = 15 # exclude first points to remove artifacts
#
#     # read in data
#     f = glob.glob(os.path.join(FLAGS.dir,'*.csv'))
#
#     print '\n{} files found.\n'.format(len(f))
#
#     # init axes
#     fig, cur_ax         = plt.subplots(1,1)
#     alpha, alpha_step   = 0, 1.0/len(f) # how fast to increase transparency
#
#     # threshold
#     cur_ax.axhline(threshold, linewidth=SETTINGS['linewidth'], color=[0.1,0.1,0.1])
#
#     for i, file in enumerate(sorted(f)):
#
#         # verbose
#         file_name = os.path.split(file)[-1]
#
#         # get % labels
#         start       = file_name.find('labels') + 7
#         end         = file_name.find('-summaries')
#         file_name   =  file_name[start:end]
#
#         # increase color intensity
#         alpha += alpha_step
#
#         # data frame: columns ['Wall time', 'Step', 'Value']
#         data = pd.read_csv(file)
#
#         # smoothed y data
#         X = list(data['Step'][offset:T+offset])
#
#         if FLAGS.smooth:
#             #Y = smooth(data['Value'], n_points)[:T]
#             Y = smooth_conv(data['Value'], window_len=window_len, window=FLAGS.smooth_method)[offset:T+offset]
#         else:
#             Y = data['Value'][:T]
#
#         # compute # steps to treshold
#         n_steps = np.where(Y < threshold)[0]
#         n_steps = n_steps[0] if n_steps != [] else -1
#
#         # add to plot
#         try:
#             cur_ax.axvline(X[n_steps], linewidth=SETTINGS['linewidth']/2, linestyle='--', color=SETTINGS['color'],alpha=alpha)
#             cur_ax.scatter(X[n_steps],Y[n_steps],20,color=SETTINGS['color'],alpha=alpha)
#         except:
#             pass
#
#         if file_name == '0.00':
#           cur_ax.plot(X, Y,
#                 linewidth=SETTINGS['linewidth'],
#                 color=SETTINGS['dram_color'],
#                 alpha=1.0,
#                 label='DRAM')
#
#           base_speed = n_steps
#
#           cur_ax.axvline(X[n_steps], linewidth=SETTINGS['linewidth']/2, linestyle='--', color=SETTINGS['dram_color'],alpha=1.0)
#           cur_ax.scatter(X[n_steps],Y[n_steps],20,color=SETTINGS['dram_color'],alpha=1.0)
#         else:
#             cur_ax.plot(X, Y,
#                         linewidth=SETTINGS['linewidth'],
#                         color=SETTINGS['color'],
#                         alpha=alpha,
#                         label='L-DRAM, ' + str(float(file_name)*100) + '%')
#
#         # print number of steps
#         #print '{}\t{}\tsteps to {}\t\tx{} faster'.format(file_name, n_steps, threshold, float(base_speed)/n_steps)
#
#
#
#     # save plot
#     #cur_ax.set_ylim([0,1])
#     cur_ax.set_xlabel('Nr. updates')
#     cur_ax.set_ylabel('Cross entropy (training)')
#
#     plt.legend()
#     #plt.grid('on')
#     plt.savefig('th={}'.format(threshold) + FLAGS.plot_name, bbox_inches='tight')
#     plt.close()

