"""
Given a folder with .csv files containing tensorboard data
plots relative speed up for all models and thresholds.
"""
#import matplotlib
#matplotlib.use('Agg')

# base
import  os
import  re
import  argparse
import  sys
import  glob
import  numpy               as np
import  pandas              as pd
import  matplotlib.pyplot   as plt
from mpl_toolkits.mplot3d   import Axes3D
from matplotlib import cm

import  seaborn             as sns

sns.set_style("white")

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def RSU(folder, n=10):
    """Given a folder of .csv files returns 3 speed-up metric (n_p, rsu, delta_t)

    Args:
        folder (str):   folder of .csv files
        n (int):        # thresholds to use

    Returns:
        (dict):         key: fname, value: (n_p, rsu, delta_t)
    """

    thresholds  = np.linspace(.5,0.99,n)
    T           = -1  # how many steps to show
    n_points    = 100 # conv window

    # store results
    RSU                 = {}
    base_speed          = np.empty(n) # # of examples same size as thresholds
    base_speed_steps    = np.empty(n) # # of examples same size as thresholds
    base_time           = np.empty(n) # # of examples same size as thresholds

    # read in data
    f = glob.glob(os.path.join(FLAGS.dir,'*accuracy.csv'))

    for i, file in enumerate(sorted(f)):

        # verbose
        file_name = os.path.split(file)[-1]

        # get % labels
        start       = file_name.find('labels=') + 7
        end         = file_name.find('_accuracy')
        file_name   = file_name[start:end]
        print '\n-- {} --'.format(file_name)


        # create dict entry
        RSU[file_name] = ([], [], [], [], [])

        # data frame: columns ['wall time', 'step', 'value']
        data = pd.read_csv(file)

        # smoothed y data
        X           = list(data['step'][:T])
        walltime    = list(data['wall_time'][:T])

        # anchors
        start_time  = walltime[0]
        N           = X[-1]*FLAGS.batch_size # examples
        print N


        if FLAGS.smooth:
            Y = smooth(data['value'][:T], n_points)
        else:
            Y = data['value'][:T]

        for k,th in enumerate(thresholds):

            # compute # steps to treshold
            n_steps = np.where(Y > th)[0]
            n_steps = n_steps[0] if n_steps != [] else -1

            if file_name == '0.00' or file_name == '0.0':
              base_speed[k]   = X[n_steps]+1
              base_time[k]    = abs(start_time - walltime[n_steps])

            # store speed-up metrics
            n_p     = (X[n_steps]+1) * FLAGS.batch_size    # examples
            t_p     = abs(start_time - walltime[n_steps])  # training time
            d_t     = (base_time[k] - t_p) / 60.**2  # difference in training time
            rsu     = base_speed[k] / float(X[n_steps]+1) if n_steps != 0 else np.nan

            RSU[file_name][0].append(n_p)
            RSU[file_name][1].append(t_p / 60.**2)
            RSU[file_name][2].append(d_t)
            RSU[file_name][3].append(rsu)
            RSU[file_name][4].append(1 - float(t_p) / base_time[k])

            print('threshold: {:.2f}\t{:6d} examples ({:<5.2f}%)\t{:>.2f}x SU\twalltime: {:<5.4f}\tdtime: {:>.2f}\tftime: {:>.2f}'.format(
                th, n_p, n_p * 100 / float(N), rsu, t_p / 60.**2, d_t, 1 - float(t_p) /base_time[k]  ))


    return thresholds, RSU

if __name__ == '__main__':

    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir','-d', type=str, default=None,
                        help='Data directory of .csv files.')
    parser.add_argument('--plot_name','-p', type=str, default='RSU.pdf',
                        help='Name of resulting plot')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--threshold','-th', type=float, default=0.9,
                        help='Threshold to count towards.')
    parser.add_argument('--smooth', default=False, action='store_true',
                        help='True: smooth, False: do not.')
    parser.add_argument('--metric', type=int, default=0,
                        help="Which metric: ['n_p', 't_p', '$\delta RWT$', 'RSU']")
    FLAGS, _ = parser.parse_known_args()
    # ------------------------------

    # parameters
    D = 2 # plot dimensionality
    n = 15
    th, rsu = RSU(FLAGS.dir, n=n)

    metric   = FLAGS.metric # 0 - 3
    metrics  = ['n_p', 't_p', '$\delta RWT$', 'RSU', 'fT']
    fig = plt.figure()

    if D == 3:
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()

    # alpha
    #alpha, alpha_step   = 0, 1.0/len(rsu.keys()) # how fast to increase transparency

    # get line colors
    cm_subsection = np.linspace(0, 1, len(rsu.keys()))
    colors = [ cm.Reds(x) for x in cm_subsection ]

    # mark fixed threshold
    ax.axvline(85, color=[.7,.7,.7], linestyle='--')

    for i, model in enumerate(sorted(rsu.keys())):

        model_float = float(model)
        x           = np.tile(model_float, th.size)

        # plot
        #alpha += alpha_step

        if model == '0.00' or model == '0.0':
            #pass
            if D == 3:
                ax.plot(x, th*100, rsu[model][metric], label=model, color='blue')
            else:
                if metric == 0 or metric == 1:
                    ax.semilogy(th*100, rsu[model][metric], label=model, color='blue')
        else:
            if D == 3:
                ax.plot(x, th*100, rsu[model][metric], label=model, color=colors[i])
            else:
                ax.plot(th*100, rsu[model][metric], label=model, color=colors[i])


    if D == 3:
        ax.set_xlabel('% labels')
        ax.set_ylabel('threshold (% accuracy)')
        ax.set_zlabel(metrics[metric])
    else:
        ax.set_xlabel('threshold (% accuracy)')
        ax.set_ylabel(metrics[metric])
        ax.set_yticks([1e5,1e6])

    plt.legend()

    if D==3:
        plt.show()

    plt.savefig(os.path.join(FLAGS.dir, metrics[metric] + FLAGS.plot_name))
    #exit()
    # histograms of RSU
    # fig, ax = plt.subplots()
    #
    # cm_subsection = np.linspace(0, 1, 5)
    # colors = [ cm.Reds(x) for x in cm_subsection ]
    #
    # for i, model in enumerate(sorted(rsu.keys())):
    #
    #     model_float = float(model)
    #
    #     if i > 0:
    #         alpha += alpha_step
    #
    #         y = rsu[model][metric]
    #         y = [b for b in y if not np.isnan(b)]
    #         plt.hist(y, bins=35, label=model, color=colors[i])
    #
    #
    # plt.legend()
    # plt.xlabel(metrics[metric])
    # plt.ylabel('Frequency')
    # plt.savefig(metrics[metric] + '_hist_' + FLAGS.plot_name)

    fig, ax = plt.subplots()

    cm_subsection = np.linspace(0, 1, 6)
    colors = [ cm.Reds(x) for x in cm_subsection ]

    k,l=[],[]

    for i, model in enumerate(sorted(rsu.keys())):

        model_float = float(model)

        if i > 0:
            #alpha += alpha_step

            y = rsu[model][metric]
            y = [b for b in y if not np.isnan(b)]

            k.append(model_float)
            l.append(np.mean(y))

            ax.plot(k, l, '--', color='b')
            ax.scatter(k, l, 100, color='b')
            #plt.errorbar(model_float, np.mean(y), yerr=np.std(y), color='r', fmt='o')



    #plt.legend()
    plt.xlabel('% labels')
    plt.ylabel('Mean {} across thresholds'.format(metrics[metric]))
    plt.savefig(os.path.join(FLAGS.dir, metrics[metric] + '_scatter_' + FLAGS.plot_name))
