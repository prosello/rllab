import argparse
import os.path as osp
import numpy as np
import csv
import matplotlib.pyplot as plt
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
    args = parser.parse_args()

    plot_experiments(args.logfiles)
    plt.show()

if __name__ == '__main__':
    main()
    
