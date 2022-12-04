import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """Load generic pickle file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data


def save_pickle(file_obj, filename):
    """Write as pickle to file"""
    with open(filename, 'wb') as f:
        pickle.dump(file_obj, f)


def generate_posterior_histograms(fit_obj, param_list, prefix=''):
    """Create and save marginal histograms for each parameter from sampled posterior"""
    n_params = len(param_list)

    fig, ax = plt.subplots(n_params, 1, figsize=(8, 12))
    plt.tight_layout()

    for j in range(n_params):
        ax[j].hist(fit_obj[param_list[j]].flatten().tolist(), bins=50)
        ax[j].set_xlabel(param_list[j])
    outfile = prefix + 'sampled_histogram.png'
    plt.savefig(os.path.join('figs', outfile))
    plt.show(); plt.close()


def generate_traceplots(fit_obj, param_list, prefix=''):
    """Create and save traceplots for each parameter from sampling algorithm"""
    n_params = len(param_list)
    n_samples = len(fit_obj[param_list[0]][0])

    fig, ax = plt.subplots(n_params, 1, sharex=True, figsize=(8, 11))
    plt.tight_layout()

    for j in range(n_params):
        ax[j].scatter(np.linspace(0, n_samples, num=n_samples), fit_obj[param_list[j]])
        ax[j].set_ylabel(param_list[j])
    plt.xlabel('number of samples')
    outfile = prefix + 'sampled_traceplot.png'
    plt.savefig(os.path.join('figs', outfile))
    plt.show(); plt.close()
