import os
import sys

import stan
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from common import load_data, save_pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from common import generate_posterior_histograms, generate_traceplots

MODEL_ID = 2


def create_summary_dfs(data):
    """Outputs list of summary data frames for each battery cell"""
    output_df = []
    for bc in data.keys():
        tmp = pd.DataFrame(data[bc]['summary'])
        output_df.append(tmp)
    return output_df


def create_features(dfs):
    """[PLACEHOLDER FUNCTION FOR KARTHIK"""
    x1_nominal_capacity = []
    x2_average_temperature = []
    x3_qd_pct_change_first100 = []

    for df in dfs:
        x1_nominal_capacity.append(df.loc[1,'QD'])
        x2_average_temperature.append(df.loc[1,'Tavg'])
        x3_qd_pct_change_first100.append((df.loc[100,'QD']-df.loc[1,'QD'])/df.loc[1,'QD'])

    return pd.DataFrame({
        'x1': np.array(x1_nominal_capacity),
        'x2': np.array(x2_average_temperature),
        'x3': np.array(x3_qd_pct_change_first100)
    })


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


def get_pred(data, params, model_id, scale=1000):
    """Helper function to access predictions for discharge capacity"""
    cycle_life = int(data['cycle_life'][0]) - 2
    y_true = np.array(data['summary']['QD'][1:(cycle_life+1)])
    x = np.linspace(0, cycle_life, cycle_life) / scale  # / cycle_life
    y_pred = fetch_model(x, params, model_id)

    return y_pred, y_true


def get_rul(data, params, model_id, threshold=0.8):
    """Helper function to access prediction for remianing useful life"""
    cycle_life = int(data['cycle_life'][0]) - 2
    nominal = data['summary']['QD'][1]
    y_val = nominal.copy()
    cycle_count = 0
    while (y_val / nominal) > threshold:
        cycle_count += 1
        y_val = fetch_model(cycle_count, params, model_id)
        # print(y_val)
    y_pred = cycle_count
    y_true = cycle_life

    return y_pred, y_true


def evaluate_fit(y_true, params, model_id, start_idx=100):
    """Compute mse for each battery cell curve"""
    i = 0
    mse_store = []; rul_mape_store = []
    for bc in data.keys():
        params_bc = [params[0][i], params[1][i], params[2][i]]
        y_pred, y_true = get_pred(data[bc], params_bc, model_id)
        rul_pred, rul_true = get_rul(data[bc], params_bc, model_id)
        mse = mean_squared_error(y_pred[start_idx:], y_true[start_idx:])
        rul_mape = mean_absolute_percentage_error(np.array([rul_pred]), np.array([rul_true]))
        mse_store.append(mse); rul_mape_store.append(rul_mape)

        i += 1

    return mse_store, rul_mape_store


def plot_examples(data, ex_list, params, model_id, scale=1000):
    """Plot y_true vs y_pred for specified alpha, beta, gamma"""
    i = 0
    for bc in data.keys():
        if bc in ex_list:
            params_bc = [params[0][i], params[1][i], params[2][i]]
            cycle_life = int(data[bc]['cycle_life'][0])-2
            y_pred, y_true = get_pred(data[bc], params_bc, model_id)
            x = np.linspace(0, cycle_life, cycle_life) / scale

            plt.scatter(x, y_true, color='grey')
            plt.plot(x, y_pred, color='r')
            plt.ylim((0.8, 1.2))

            outfile = 'bayes_plot_' + bc + '.png'
            plt.savefig(os.path.join('figs', outfile))
            plt.show(); plt.close()
        i += 1


def prepare_data_for_stan(dfs):
    """Create dictionary of data inputs for stan model"""
    y = []
    N_BC = []

    for df in dfs:
        N_BC.append(df.shape[0])
        y.append(np.array(df['QD']))
    X = create_features(dfs)  # check dim!
    y = np.hstack(y)

    return {
        'T': np.sum(N_BC),               # total number of cycles
        'N': len(dfs),
        'd': X.shape[1],
        'y': y,                          # flattened labels
        'x1': np.array(X['x1']),
        'x2': np.array(X['x2']),
        'x3': np.array(X['x3']),
        # 'X': X.to_json(),                # feature variables
        'N_BC': np.array(N_BC)
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
        vector[N] x1;
        vector[N] x2;
        vector[N] x3;
        array[N] int N_BC;          // cycle life for each battery cell
    }
    parameters {
        real alpha_0;
        real alpha_1;
        real alpha_2;
        real alpha_3;

        real beta_0;
        real beta_1;
        real beta_2;
        real beta_3;

        //real gamma_0;
        //real gamma_1;
        //real gamma_2;
        //real gamma_3;

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
            alpha[i] = alpha_0 + alpha_1*x1[i] + alpha_2*x2[i] + alpha_3*x3[i];
            beta[i] = beta_0 + beta_1*x1[i] + beta_2*x2[i] + beta_3*x3[i];
            gamma[i] = x1[i];

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
        alpha_0 ~ normal(2.5, 1);
        alpha_1 ~ normal(0, 1);
        alpha_2 ~ normal(0, 1);
        alpha_3 ~ normal(0, 1);

        beta_0 ~ normal(2.5, 1);
        beta_1 ~ normal(0, 1);
        beta_2 ~ normal(0, 1);
        beta_3 ~ normal(0, 1);

        //gamma_0 ~ normal(1.1, 1);
        //gamma_1 ~ normal(0, 1);
        //gamma_2 ~ normal(0, 1);
        //gamma_3 ~ normal(0, 1);

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


def prepare_params_given_samples(fit, stan_data):
    """Helper function to pull together mle estimates given posterior samples"""
    alpha = fit['alpha_1'].mean()*stan_data['x1'] + fit['alpha_2'].mean()*stan_data['x2'] + fit['alpha_3'].mean()*stan_data['x3'] + fit['alpha_0'].mean()
    beta = fit['beta_1'].mean()*stan_data['x1'] + fit['beta_2'].mean()*stan_data['x2'] + fit['beta_3'].mean()*stan_data['x3'] + fit['beta_0'].mean()
    gamma = stan_data['x1']

    return [alpha, beta, gamma]


if __name__ == '__main__':

    # load data
    infile = sys.argv[1]
    data = load_data(os.path.join('data', infile))
    train = pd.read_csv(os.path.join('data', 'train.csv'))
    test = pd.read_csv(os.path.join('data', 'test.csv'))
    dfs = create_summary_dfs(data)

    # bayes model
    # print(test_script_for_stan())
    stan_code = prepare_code_for_stan()
    stan_data = prepare_data_for_stan(dfs)
    posterior = stan.build(stan_code, data=stan_data, random_seed=101)
    fit = posterior.sample(num_samples=1000, num_chains=1)
    save_pickle(posterior, filename='model.pkl')
    save_pickle(fit, filename='fit.pkl')

    # evaluate fit
    map = {
        1: (0.2, 2, 0.4),  # alpha, beta, gam
        2: (2.5, 2.5, 1.1),  # shape, midpoint, asymptote
    }
    params = prepare_params_given_samples(fit, stan_data)  # params = map[MODEL_ID]
    mse_store, rul_mape_store = evaluate_fit(data, params=params, model_id=MODEL_ID)

    # write results
    param_list = [
        'alpha_0',
        'alpha_1',
        'alpha_2',
        'alpha_3',
        'beta_0',
        'beta_1',
        'beta_2',
        'beta_3',
        'sigma'
    ]
    generate_posterior_histograms(fit, param_list)
    generate_traceplots(fit, param_list)

    print('MSE for Discharge Capacity: {}'.format(np.mean(mse_store)))
    print('MAPE for Remaining Useful Life: {}'.format(np.mean(rul_mape_store)))
    plot_examples(data, params=params, model_id=MODEL_ID, ex_list=['b1c0', 'b1c1'])
