import os
import sys

import stan
import arviz
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from utils.common import load_data, save_pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from utils.common import generate_posterior_histograms, generate_traceplots

MODEL_ID = 2
USE_CACHE = True


def create_summary_dfs(data):
    """Outputs list of summary data frames for each battery cell"""
    output_df = []
    for bc in data.keys():
        tmp = pd.DataFrame(data[bc]['summary'])
        output_df.append(tmp)
    return output_df


def exp_decay_function(x, params):  # alpha=0.2, beta=2, gam=0.4; # alpha=0.000005, beta=1.44, gam=15000
    """Functional form of exponential decay, shape and translation"""
    alpha, beta, gam = params
    return 2.1 - np.exp(alpha*(np.power(x, beta)-gam))


def inv_sigmoid_function(x, params):  # shape=2, midpoint=2, asymptote=1.1
    """Functional form of shifted inverse sigmoid"""
    # Note, extra parameter allows more freedom than exponential decay
    shape, midpoint, asymptote = params
    return asymptote - (1 / (1 + np.exp(-shape * (x - midpoint))))


def fetch_model(x, params, model_id):
    """Helper function to fetch specified basis function"""
    if model_id == 1:  # exponential decay
        y_pred = exp_decay_function(x, params)

    elif model_id == 2:  # inverse sigmoid
        y_pred = inv_sigmoid_function(x, params)

    else:
        print("Error: model id not correctly specified")

    return y_pred


def get_pred(num_cycles, params, model_id, scale=1000):
    """Helper function to access predictions for discharge capacity"""
    n_samples = params[0].shape[0]
    params_preproc = [[params[0][i], params[1][i], params[2][i]] for i in range(n_samples)]

    x = np.linspace(0, num_cycles, num_cycles) / scale  # / cycle_life
    # y_pred = fetch_model(x, params, model_id)
    y_pred = np.array([fetch_model(x, params_preproc[i], model_id) for i in range(n_samples)])

    return y_pred


def get_rul(threshold, params, model_id, scale=1000):
    """Helper function to access prediction for remianing useful life"""
    y_val = params[2][0]  # nominal
    count = 0
    while y_val > threshold:
        count += 1
        scaled_x = count / scale
        y_vals = fetch_model(scaled_x, params, model_id)
        y_val = np.median(y_vals, axis=0)
    y_pred = count / scale

    return y_pred


def evaluate_fit(y_test, params, model_id, start_idx=100, scale=1000):
    """Compute mse for each battery cell curve"""
    mse_store = []; rul_mape_store = []
    for i in range(len(y_test)):
        cycle_life, threshold = len(y_test[i]), y_test[i][-1]
        params_i = [params[0][i], params[1][i], params[2][i]]

        y_true, rul_true = y_test[i], cycle_life/scale
        y_pred = np.median(get_pred(cycle_life, params_i, model_id), axis=0)
        rul_pred = get_rul(threshold, params_i, model_id)

        mse = mean_squared_error(y_pred[start_idx:], y_true[start_idx:])
        rul_mape = mean_absolute_percentage_error(np.array([rul_pred]), np.array([rul_true]))
        mse_store.append(mse); rul_mape_store.append(rul_mape)

    return mse_store, rul_mape_store


def plot_predicted_curve(y_test, test_bat_ids, params, model_id, num_plots=1, scale=1000):
    """Plot y_true vs y_pred for specified alpha, beta, gamma"""
    plt.rcParams.update({'font.size': 14})
    for i, id in enumerate(test_bat_ids):
        if i >= num_plots:
            break

        y_true = y_test[i]
        cycle_life = len(y_true)  # y_test_i[-1]
        params_i = [params[0][i], params[1][i], params[2][i]]

        y_pred = np.median(get_pred(cycle_life, params_i, model_id), axis=0)
        x = np.linspace(0, cycle_life, cycle_life) / scale
        # x2 = np.linspace(0, 2*cycle_life, cycle_life) / scale

        plt.scatter(x, y_true, color='grey', s=0.75)
        plt.plot(x, y_pred, color='r')
        plt.ylim((0.8, 1.2))
        # plt.ylim((0, 1.2))

        # plt.title(r'$\gamma - 1 / (1+\exp(-\alpha(x-\beta)))$')
        # plt.title(r'$\gamma - exp(\alpha(x-\beta))$')
        plt.ylabel('Discharge capacity (Qd) for {}'.format(id))
        plt.xlabel('Number of cycles / 1000')

        # outfile = 'example_inv_sigmoid.png'
        # outfile = 'example_exponetial_decay.png'
        outfile = 'bayes_plot_' + id + '.png'
        plt.savefig(os.path.join(os.path.pardir, 'figs', outfile))
        plt.show(); plt.close()


def plot_predicted_curve_with_error(y_test, test_bat_ids, params, model_id, num_plots=10, scale=1000):
    """Plot y_true vs y_pred for specified alpha, beta, gamma"""
    plt.rcParams.update({'font.size': 14})
    for i, id in enumerate(test_bat_ids):
        if i >= num_plots:
            break

        y_true = y_test[i]
        cycle_life = len(y_true)  # y_test_i[-1]
        params_i = [params[0][i], params[1][i], params[2][i]]

        y_preds = get_pred(cycle_life, params_i, model_id)
        y_pred_median = np.median(y_preds, axis=0)
        y_pred_low, y_pred_high = np.quantile(y_preds, [0.2, 0.8], axis=0) # y_pred_median - 2*np.std(y_preds, axis=0)

        x = np.linspace(0, cycle_life, cycle_life) / scale
        # x2 = np.linspace(0, 2*cycle_life, cycle_life) / scale

        plt.scatter(x, y_true, color='grey', s=0.75)
        plt.fill_between(x, y_pred_low, y_pred_high, alpha=0.3, color='blue')
        plt.plot(x, y_pred_median, color='r')
        plt.ylim((0.8, 1.2))
        # plt.ylim((0, 1.2))

        # plt.title(r'$\gamma - 1 / (1+\exp(-\alpha(x-\beta)))$')
        # plt.title(r'$\gamma - exp(\alpha(x-\beta))$')
        plt.ylabel('Discharge capacity (Qd) for {}'.format(id))
        plt.xlabel('Number of cycles / 1000')

        # outfile = 'example_inv_sigmoid.png'
        # outfile = 'example_exponetial_decay.png'
        outfile = 'bayes_plot_with_error_train_' + id + '.png'
        plt.savefig(os.path.join(os.path.pardir, 'figs', outfile))
        plt.show(); plt.close()


def create_features(train_data):
    """[PLACEHOLDER FUNCTION FOR KARTHIK"""
    # print(train_data.groupby(['bat_id']).last().index)
    X_df = train_data.groupby(['bat_id']).last().reset_index(drop=True)
    X_df.columns = X_df.columns.str.lower()

    x1_nominal_capacity = train_data.groupby(['bat_id']).nth(5)['QD']
    x2_variance_100v10 = X_df['variance'].copy()
    x3_log_min_difference = X_df['log|min(delta(q(v)))|'].copy()
    x4_chargetime_average = X_df['chargetimeavg cyc1-5'].copy()
    x5_avg_temperature_integral = X_df['temp integral'].copy()
    # x6_intercept_2to100 = X_df['intercept 2-100'].copy()

    X = pd.DataFrame({
        'x1': np.array(x1_nominal_capacity),
        'x2': np.array(x2_variance_100v10),
        'x3': np.array(x3_log_min_difference),
        'x4': np.array(x4_chargetime_average),
        'x5': np.array(x5_avg_temperature_integral),
        # 'x6': np.array(x6_intercept_2to100),
    })

    X_scaled = scale(X)
    X_scaled[:, 0] = np.array(x1_nominal_capacity)  # overwrite nominal capacity
    return X_scaled


def prepare_data_for_stan(X_scaled, train_y):
    """Create dictionary of data inputs for stan model"""
    labels = np.hstack(train_y)
    N_BC = np.array([len(y_i) for y_i in train_y])

    return {
        'T': np.sum(N_BC),               # total number of cycles
        'N': len(train_y),
        'd': X_scaled.shape[1],
        'y': labels,                          # flattened labels
        'X': X_scaled,
        'N_BC': N_BC
    }


def prepare_code_for_stan():
    """Specify model code"""
    model_code = """
    functions {
        real exponential_decay(
            real x,
            real alpha,
            real beta,
            real gamma
        ) {
            return 2 - exp(alpha*(pow(x, beta) - gamma));
        }
        real inv_sigmoid(
            real x,
            real alpha,
            real beta,
            real gamma
        ) {
            return gamma - (1 / (1 + exp(-alpha * (x - beta))));
        }
    }
    data {
        int<lower=0> T;             // total cycles observed
        int<lower=0> N;             // number of battery cells = 200
        int<lower=0> d;             // number of feature dimensions
        vector[T] y;
        matrix[N,d] X;
        array[N] int N_BC;          // cycle life for each battery cell
    }
    parameters {
        real a_0, a_1, a_2, a_3, a_4, a_5;
        real b_0, b_1, b_2, b_3, b_4, b_5;
        //real g_0, g_1;

        real<lower=0> sigma;
    }
    transformed parameters {
        vector[N] alpha;
        vector[N] beta;
        vector[N] gamma;
        vector[T] y_hat;
    {
        int idx = 1;
        real scaled_cycle_count;

        for(i in 1:N) {
            alpha[i] = a_0 + a_1*X[i,1] + a_2*X[i,2] + a_3*X[i,3] + a_4*X[i,4] + a_5*X[i,5];
            beta[i] = b_0 + b_1*X[i,1] + b_2*X[i,2] + b_3*X[i,3] + b_4*X[i,4] + b_5*X[i,5];
            gamma[i] = X[i,1];  // first few entries have measurment error

            for (j in 1:N_BC[i]) {
                scaled_cycle_count = j / 1000.0;
                //y_hat[idx] = exponential_decay(scaled_cycle_count, alpha[i], beta[i], gamma[i]);
                y_hat[idx] = inv_sigmoid(scaled_cycle_count, alpha[i], beta[i], gamma[i]);
                idx += 1;
            }
        }
    }
    }
    model {
        a_0 ~ normal(5, 1);
        a_1 ~ normal(0, 1);
        a_2 ~ normal(0, 1);
        a_3 ~ normal(0, 1);
        a_4 ~ normal(0, 1);
        a_5 ~ normal(0, 1);

        b_0 ~ normal(1, 1);
        b_1 ~ normal(0, 1);
        b_2 ~ normal(0, 1);
        b_3 ~ normal(0, 1);
        b_4 ~ normal(0, 1);
        b_5 ~ normal(0, 1);

        //g_0 ~ normal(1.1, 1);
        //g_1 ~ normal(0, 1);

        sigma ~ gamma(1, 2);

        y ~ normal(y_hat, sigma);
    }
    """
    return model_code


def test_script_for_stan():
    model = """
    data {
        int<lower=0> N;
        vector[N] x;
        vector[N] y;
    }
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    model {
        y ~ normal(alpha + beta * x, sigma);
    }
    """

    # parameters to be inferred
    alpha = 4.0
    beta = 0.5
    sigma = 1.0

    # generate and plot data
    x = 10 * np.random.rand(100)
    y = alpha + beta * x
    y = np.random.normal(y, scale=sigma)

    # put our data in a dictionary
    stan_data = {'N': len(x), 'x': x, 'y': y}

    # compile the model
    posterior = stan.build(model, data=stan_data, random_seed=101)

    # train the model and generate samples
    fit = posterior.sample(num_samples=1000, num_chains=4)
    fit['alpha']

    return "Test script runs without error!"


def prepare_params_given_samples(fit, X):
    """Helper function to pull together mle estimates given posterior samples"""
    x_1, x_2, x_3, x_4, x_5 = [X[:, i].reshape(1,-1) for i in range(X.shape[1])]

    a_0, a_1, a_2, a_3, a_4, a_5 = fit['a_0'], fit['a_1'], fit['a_2'], fit['a_3'], fit['a_4'], fit['a_5']
    b_0, b_1, b_2, b_3, b_4, b_5 = fit['b_0'], fit['b_1'], fit['b_2'], fit['b_3'], fit['b_4'], fit['b_5']
    # g_0, g_1 = fit['g_0'], fit['g_1']

    alpha = a_0 + x_1.T@a_1 + x_2.T@a_2 + x_3.T@a_3 + x_4.T@a_4 + x_5.T@a_5
    beta = b_0 + x_1.T@b_1 + x_2.T@b_2 + x_3.T@b_3 + x_4.T@b_4 + x_5.T@b_5
    gamma = x_1.T  # g_0 + g_1*x_1

    return [alpha, beta, gamma]


def prepare_label_data(data_dict, train_ids, test_ids):
    """Helper function to prepare labels for train and test"""
    train_y = []
    for id in train_ids:
        train_y.append(data_dict[id]['summary']['QD'][1:])

    test_y = []
    for id in test_ids:
        test_y.append(data_dict[id]['summary']['QD'][1:])

    # print("Error: we are missing some battery ids!")

    return train_y, test_y


if __name__ == '__main__':

    # load data
    infile = sys.argv[1]
    data = load_data(os.path.join(os.path.pardir, 'data', infile))
    train_dat = pd.read_csv(os.path.join(os.path.pardir, 'data', 'train.csv'))
    test_dat = pd.read_csv(os.path.join(os.path.pardir, 'data', 'test.csv'))

    train_bat_ids = sorted(train_dat['bat_id'].unique())
    test_bat_ids = sorted(test_dat['bat_id'].unique())
    y_train, y_test = prepare_label_data(data, train_bat_ids, test_bat_ids)

    # bayes model
    if USE_CACHE:
        # 0: dummy data, 1: original formulation, 2: added hyperprior
        fit = load_data(os.path.join('data', 'fit_1.pkl'))

    else:
        # print(test_script_for_stan())
        stan_code = prepare_code_for_stan()
        X_train = create_features(train_dat)
        stan_data = prepare_data_for_stan(X_train, y_train)
        posterior = stan.build(stan_code, data=stan_data, random_seed=101)
        fit = posterior.sample(num_samples=1000, num_chains=1)
        # save_pickle(posterior, filename='model_1.pkl')
        save_pickle(fit, filename='fit_new.pkl')

    # evaluate fit
    n = len(y_test)
    map = {
        1: [np.ones((n,10))*0.2, np.ones((n,10))*2.1, np.ones((n,1))*0.1],  # alpha, beta, gam
        2: [np.ones((n,10))*3, np.ones((n,10))*1.4, np.ones((n,1))*1.1],  # shape, midpoint, asymptote
    }
    X_test = create_features(test_dat)
    # params = map[MODEL_ID]
    # params = prepare_params_given_samples(fit, X_test)
    params = [fit['alpha'], fit['beta'], fit['gamma']]
    mse_store, rul_mape_store = evaluate_fit(y_train, params=params, model_id=MODEL_ID)  # y_test

    # Autocorrelation
    sample_arr = np.array(fit.to_frame()[[fit.param_names[i] for i in range(13)]])
    ess = arviz.ess(arviz.convert_to_dataset(sample_arr.reshape(1,-1,13)))
    print('Number of effective samples: {}'.format(ess.mean()))

    # write results
    param_list_alpha = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5']
    generate_posterior_histograms(fit, param_list_alpha, prefix='bayes_alpha_')
    generate_traceplots(fit, param_list_alpha, prefix='bayes_alpha_')

    param_list_beta = ['b_0', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5']
    generate_posterior_histograms(fit, param_list_beta, prefix='bayes_beta_')
    generate_traceplots(fit, param_list_beta, prefix='bayes_beta_')

    print('MSE for Discharge Capacity: {}'.format(np.mean(mse_store)))
    print('MAPE for Remaining Useful Life: {}'.format(np.mean(rul_mape_store)))
    # plot_predicted_curve(y_test, test_bat_ids, params=params, model_id=MODEL_ID)  # test_bat_ids
    plot_predicted_curve_with_error(y_train, train_bat_ids, params=params, model_id=MODEL_ID)
