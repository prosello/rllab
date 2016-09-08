#!/usr/bin/env python
import argparse
import os.path as osp
import numpy as np
import pandas as pd
import json
from glob import glob


def plot_experiments(files, legend=False, post_processing=None, key='AverageReturn'):
    if not isinstance(files, (list, tuple)):
        files = [files]
    print 'plotting the following experiments:'
    for f in files:
        print f
    plots = []
    legends = []
    for f in files:
        exp_name = osp.basename(f)
        returns = []
        with open(f, 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row[key]:
                    returns.append(float(row[key]))
        returns = np.array(returns)
        if post_processing:
            returns = post_processing(returns)
        plots.append(plt.plot(returns)[0])
        legends.append(exp_name)
    if legend:
        plt.legend(plots, legends)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfiles', type=str, nargs='+')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--fields', type=str, default='all')
    parser.add_argument('--plotfile', type=str, default=None)
    args = parser.parse_args()

    assert len(set(args.logfiles)) == len(args.logfiles), 'Log files must be unique'

    fname2log = {}
    for fname in args.logfiles:
        df = pd.read_csv(fname)
        if 'Iteration' in df.keys():
            df.set_index('Iteration', inplace=True)
        elif 'Epoch' in df.keys():
            df.set_index('Epoch', inplace=True)
        else:
            raise NotImplementedError()
        if not args.fields == 'all':
            df = df.loc[:, args.fields.split(',')]
        fname2log[fname] = df

    if not args.noplot or args.plotfile is not None:
        import matplotlib
        if args.plotfile is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-colorblind')

    ax = None
    for fname, df in fname2log.items():
        with pd.option_context('display.max_rows', 9999):
            print(fname)
            print(df[-1:])

        if not args.noplot:
            if ax is None:
                ax = df.plot(subplots=True, title=','.join(args.logfiles))
            else:
                df.plot(subplots=True, title=','.join(args.logfiles), ax=ax, legend=False)
    if args.plotfile is not None:
        plt.savefig(args.plotfile, transparent=True, bbox_inches='tight', dpi=300)
    elif not args.noplot:
        plt.show()

    # plot_experiments(args.logfiles)
    # plt.savefig('/tmp/progress.png')
    # plt.show()


if __name__ == '__main__':
    main()
