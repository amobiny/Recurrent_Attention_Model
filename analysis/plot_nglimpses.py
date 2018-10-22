# base
import  os
import  re
import  argparse
import  sys
import  glob
import  cPickle as pickle
import  numpy               as np
import  pandas              as pd
import  matplotlib.pyplot   as plt
from mpl_toolkits.mplot3d   import Axes3D
from matplotlib import cm

import  seaborn             as sns
sns.set_style("white")

if __name__ == '__main__':

    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir','-d', type=str, default=None,
                        help='Directory of saved dicts.')
    parser.add_argument('--plot_name','-p', type=str, default='nglimpses.pdf',
                        help='Name of resulting plot')
    FLAGS, _ = parser.parse_known_args()
    # ------------------------------

    # load dicts
    f = glob.glob(os.path.join(FLAGS.dir,'*.p'))

    # line colors
    cm_subsection = np.linspace(0, 1, len(f))
    colors = [ cm.Reds(x) for x in cm_subsection ]

    fig, ax = plt.subplots()

    for i, file in enumerate(sorted(f)):

        # % labels
        print file
        fname = file.split('_')[0]

        d = pickle.load(open(FLAGS.file, 'rb'))

        # plot performance for all glimpses
        for n in d.keys():

            ax.errorbar(n, np.mean(d[n]['test']), yerror=np.std(d[n]['test']),
                        color=colors[i], labels=fname)

    plt.xlabel('Number of glimpses')
    plt.ylabel('Test error (%)')
    plt.legend()
    plt.show()
    #plt.savefig(FLAGS.plot_name, bbox_inches='tight')
