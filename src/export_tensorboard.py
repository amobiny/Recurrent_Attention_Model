"""
Saves Tensorboard summaries as .csv

Tested on TF v1.0
"""

import  os
import  re
import  warnings
import  argparse
import  pandas as pd
from    tensorflow.python.summary import event_accumulator

# ----- parse command line -----
parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default='.',
                    help='Path to event file.')
parser.add_argument('--output_dir', type=str, default='.',
                    help='Directory to save csv to.')
parser.add_argument('--prefix', type=str, default='',
                    help='Prefix to prepend to filenames.')
args, _ = parser.parse_known_args()
#--------------------------------

def load_summaries(fname, summaries=[]):
    """Loads summaries file as DataFrames.

    Args:
        fname (str):            event file name
        summaries (list):       list of summary names (saves all scalars if empty)

    Returns:
        (dict):                 key: summary name, value: DataFrame with columns ['step', 'wall_time', 'value']
    """
    #warnings.warn('Only implemented for scalar summaries.', Warning)

    if not os.path.isfile(fname):
        print('\n<<{}>> is not a file.'.format(fname))
        return
    else:
        print('\nLoading <<{}>> event file . . .'.format(fname)),

    # return dict of DataFrames
    df = {}

    # all summaries
    events = event_accumulator.EventAccumulator(fname).Reload()
    print('[OK]')

    # tags
    tags = events.Tags()

    if summaries == []:
        summaries = tags['scalars']

    for summary in summaries:
        try:
            df[summary] = pd.DataFrame(events.Scalars(summary))
            print('{:30s} [loaded]'.format(summary))
        except:
            print('{:30s} [not found]'.format(summary))

    return df

def save_summaries(df, dir='.', prefix=''):
    """ Given a dictionary of summaries saves them to .csv where
    the summary names are the file names.

    Args:
        df (dict):      key: summary name, value: DataFrame - as returned by load_summaries
        dir (str):      path where .csv files will be saved
        prefix (str):   optinally prepend string to file names, '{prefix}_{summary}.csv'
    """

    if not os.path.exists(dir):
        os.mkdir(dir)

    for sname in df.keys():

        try:
            fname = os.path.join(dir, '{}_{}.csv'.format(prefix, sname))
            df[sname].to_csv(fname)
            print('{:30s} [saved]'.format(fname))
        except:
            print('{:30s} [failed]'.format(fname))

def export_summaries(summaries=[], dir='.', expr='summaries/val', output_dir='.'):
    """Given a parent directory containing summaries exports all of them to .csv.
    Assumes each summary in subdirectory

        ./parent/sub_dir/summaries/train/summary-file

    Args:
        dir (str):              parent source directory
        expr (str):             expression to select subdirectories
        output_dir (str):       output directory
    """

    subdirs = [x for x in os.walk(dir) if expr in x[0]] # contains (subdir, dirs, files)

    for dirs in subdirs:

        # file name
        fname       = os.path.join(dirs[0], dirs[2][0])
        save_fname  = fname.split('_')[-1]
        save_fname  = save_fname.split('/')[0]

        # export summaries
        df = load_summaries(fname, summaries=summaries)
        save_summaries(df, dir=output_dir, prefix='val_' + save_fname)


if __name__ == '__main__':

    export_summaries(summaries=['accuracy'],
                     dir=args.source_dir,
                     output_dir=args.output_dir)
