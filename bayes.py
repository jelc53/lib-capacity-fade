import os
import sys

# import pystan
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from common import load_data
from sklearn.metrics import mean_squared_error 


def create_summary_dfs(data):
    """Outputs list of summary data frames for each battery cell"""
    output_df = []
    for bc in data.keys():
        tmp = pd.DataFrame(data[bc]['summary'])
        output_df.append(tmp)
    return output_df


def create_features(data):
    """xx"""
    output_dict = {}
    for bc in data.keys():
        output_dict[bc] = data[bc]['charge_policy']
    return pd.DataFrame(output_dict)


def exp_decay_function(alpha, beta, x):
    """Functional form of exponential decay, shape and translation"""
    return 2 - np.exp(alpha*(np.power(x, beta)))


def get_pred(alpha, beta, data):
    """xx"""
    cycle_life = int(data['cycle_life'][0]) - 2
    x = np.linspace(0, cycle_life, cycle_life) / cycle_life
    y_pred = exp_decay_function(alpha, beta, x)
    y_true = np.array(data['summary']['QD'][1:(cycle_life+1)])

    return y_pred, y_true


def evaluate_fit(y_true, alpha=0.1, beta=10):
    """xx"""
    mse_store = []
    for bc in data.keys():
        y_pred, y_true = get_pred(alpha, beta, data[bc])
        mse = mean_squared_error(y_pred=y_pred, y_true=y_true)
        mse_store.append(mse)
    return mse_store


def plot_examples(data, ex):
    """xx"""
    for bc in data.keys():
        if bc in ex:
            cycle_life = int(data[bc]['cycle_life'][0])-2
            y_pred, y_true = get_pred(0.1, 10, data[bc])
            x = np.linspace(0, cycle_life, cycle_life) / cycle_life
            plt.scatter(x, y_true, color='grey')
            plt.plot(x, y_pred, color='r')
            plt.ylim((0.8, 1.2))
            outfile = bc + '_example_plot.png'
            plt.savefig(os.path.join('figs', outfile))
            plt.show(); plt.close()


if __name__ == '__main__':

    infile = sys.argv[1]
    data = load_data(os.path.join('data', infile))
    dfs = create_summary_dfs(data)
    mse_store = evaluate_fit(data)

    plot_examples(data, ex=['b1c0', 'b1c1'])
