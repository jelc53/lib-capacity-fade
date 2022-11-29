import os
import sys

# import pystan
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from common import load_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def create_summary_dfs(data):
    """Outputs list of summary data frames for each battery cell"""
    output_df = []
    for bc in data.keys():
        tmp = pd.DataFrame(data[bc]['summary'])
        output_df.append(tmp)
    return output_df


def create_features(data):
    """[PLACEHOLDER FUNCTION FOR KARTHIK"""
    output_dict = {}
    for bc in data.keys():
        output_dict[bc] = data[bc]['charge_policy']
    return pd.DataFrame(output_dict)


def exp_decay_function(alpha, beta, gam, x):
    """Functional form of exponential decay, shape and translation"""
    return 2 - np.exp(alpha*(np.power(x, beta)-gam))


def get_pred(alpha, beta, gam, data):
    """Helper function to access predictions for discharge capacity"""
    cycle_life = int(data['cycle_life'][0]) - 2
    x = np.linspace(0, cycle_life, cycle_life)  # / cycle_life
    y_pred = exp_decay_function(alpha, beta, gam, x)
    y_true = np.array(data['summary']['QD'][1:(cycle_life+1)])

    return y_pred, y_true


def get_rul(alpha, beta, gam, data, threshold=0.8):
    """Helper function to access prediction for remianing useful life"""
    cycle_life = int(data['cycle_life'][0]) - 2
    nominal = data['summary']['QD'][1]
    y_val = nominal.copy()
    cycle_count = 0
    while (y_val / nominal) > threshold:
        cycle_count += 1
        y_val = exp_decay_function(alpha, beta, gam, cycle_count)
        # print(y_val)
    y_pred = cycle_count
    y_true = cycle_life

    return y_pred, y_true


def evaluate_fit(y_true, alpha=0.000005, beta=1.44, gam=15000, start_idx=100):
    """Compute mse for each battery cell curve"""
    mse_store = []; rul_mape_store = []
    for bc in data.keys():
        y_pred, y_true = get_pred(alpha, beta, gam, data[bc])
        rul_pred, rul_true = get_rul(alpha, beta, gam, data[bc])
        mse = mean_squared_error(y_pred[start_idx:], y_true[start_idx:])
        rul_mape = mean_absolute_percentage_error(np.array([rul_pred]), np.array([rul_true]))
        mse_store.append(mse); rul_mape_store.append(rul_mape)
    return mse_store, rul_mape_store


def plot_examples(data, ex_list, alpha=0.000005, beta=1.44, gam=15000):  #alpha=0.2, beta=2, gam=0.4
    """Plot y_true vs y_pred for specified alpha, beta, gamma"""
    for bc in data.keys():
        if bc in ex_list:
            cycle_life = int(data[bc]['cycle_life'][0])-2
            y_pred, y_true = get_pred(alpha, beta, gam, data[bc])
            x = np.linspace(0, cycle_life, cycle_life) / cycle_life
            plt.scatter(x, y_true, color='grey')
            plt.plot(x, y_pred, color='r')
            plt.ylim((0.8, 1.2))
            outfile = 'example_plot_' + bc + '.png'
            plt.savefig(os.path.join('figs', outfile))
            plt.show(); plt.close()


if __name__ == '__main__':

    infile = sys.argv[1]
    data = load_data(os.path.join('data', infile))
    dfs = create_summary_dfs(data)
    mse_store, rul_mape_store = evaluate_fit(data)
    print('MSE for Discharge Capacity: {}'.format(np.mean(mse_store)))
    print('MAPE for Remaining Useful Life: {}'.format(np.mean(rul_mape_store)))

    plot_examples(data, ex_list=['b1c0', 'b1c1'])
