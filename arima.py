from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import os
import six
import sys
sys.modules['sklearn.externals.six'] = six
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import pickle
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import pmdarima
from pmdarima.arima import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def create_test_split(df, split_parameter):
    '''splitting the given battery data into a train/test split'''

    TEST_SIZE = len(df.values) - 100
    train, test = df.iloc[:-TEST_SIZE], df.iloc[-TEST_SIZE:]
    return train, test, TEST_SIZE

def acf_plots(df, d):
    '''plot ACF and PACF'''

    data_diff = df.diff(periods=d).dropna()  #differentiated data

    plot_acf(np.array(data_diff), lags=40)
    plt.show()

    plot_pacf(np.array(data_diff), lags=45)
    plt.show()


def create_single_exog_var(bat_dict, TEST_SIZE, bat_id, test=False):
    '''creating one exogenous variable of correct size'''

    exog = np.array(bat_dict[bat_id]['summary']['QD'])
    if test == False:
        #plt.plot(exog)
        #plt.show()
        param = exog[:len(exog) - TEST_SIZE]
        param = param.reshape(len(exog) - TEST_SIZE, 1)
    else:
        param = exog[len(exog) - TEST_SIZE-2:]
        param = param.reshape(TEST_SIZE+2, 1)
    return param


def create_multiple_exog_vars(bat_dict, TEST_SIZE, bat_id, test=False):
    '''creating two exogenous variables of correct size and stacking them in a matrix'''

    exog1 = np.array(bat_dict[bat_id]['summary']['QD'])
    exog2 = np.array(bat_dict[bat_id]['summary']['IR'])
    if test == False:
        #plt.plot(exog1)
        #plt.show()
        params = np.vstack((exog1[:len(exog1) - TEST_SIZE], exog2[:len(exog2) - TEST_SIZE]))
        params = params.T
    else:
        params = np.vstack((exog1[len(exog1) - TEST_SIZE:], exog2[len(exog2) - TEST_SIZE:]))
        params = params.T
    return params


def run_auto_arima(train, params, d):
    '''running auto arima, returns prediction'''
    model = auto_arima(train, exogenous=params, start_p=0, start_q=0,
                       test='adf',
                       max_p=5, max_q=5,
                       d=d,
                       seasonal=False,
                       start_P=0,
                       D=None,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=False)

    prediction = model.predict(n_periods=4000, return_conf_int=False, exogenous=params)

    return prediction


def run_arima(train):
    '''running arima with fixed paramters, returns prediction'''
    model = ARIMA(order=(2, 2, 2))
    model_fit = model.fit(train)
    prediction = model.predict(n_periods=4000, return_conf_int=False)
    return prediction


def test_model(split_parameter, bat_dict, exog_bat_id, auto, plot):
    '''testing the specified arima model on all the batteries satisfying criteria. Returns evaluation metrics'''

    mse_list = np.array([])
    mape_list = np.array([])
    cycle_life_errors = np.array([])
    cycle_life_errors_percentage = np.array([])

    i = 0
    for key in bat_dict.keys():
        try:
            df = pd.DataFrame(np.array(bat_dict[key]['summary']['QD'][5:]))
            if len(df.values) < 500:
                continue
            elif len(df.values) > 1200:
                continue
            else:
                train, test, TEST_SIZE = create_test_split(df, split_parameter)
                d = pmdarima.arima.ndiffs(train)

                if auto is True:
                    prediction = run_auto_arima(train, params, d)
                else:
                    prediction = run_arima(train)

                #cut off at test size because the prediction is much longer
                a = prediction[:TEST_SIZE]
                b = df.values[len(df.values) - TEST_SIZE:]
                
                if plot is True:
                    prediction_series = pd.Series(prediction, index=test.index)
                    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                    ax.plot(df.values, label="true values")
                    ax.plot(prediction_series, label="prediction")
                    ax.fill_between(prediction_series.index)
                    plt.show()


                    plt.plot(b, label="true values")
                    plt.plot(a, label="prediction")
                    plt.legend()
                    plt.show()

                mape = mean_absolute_percentage_error(b, a)
                mse = mean_squared_error(b, a)
                mse_list = np.append(mse_list, mse)
                mape_list = np.append(mape_list, mape)

                cycle_life = len(df.values)
                cycle_life_pred = 0
                Q_nom = df.values[0]
                k = 0
                #simple check for approximate cycle life
                for value in prediction:
                    if value / Q_nom < 0.8:
                        cycle_life_pred += train.values.shape[0] + k
                        break
                    k += 1

                cycle_life_errors = np.append(cycle_life_errors, abs(cycle_life-cycle_life_pred))
                cycle_life_errors_percentage = np.append(cycle_life_errors_percentage, abs(cycle_life-cycle_life_pred)/cycle_life)
            i += 1
        except ValueError:
            print("hej")

    return mape_list, mse_list, cycle_life_errors, cycle_life_errors_percentage

if __name__ == '__main__':
    data_dict = load_data('/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/processed_data_lite.pkl')
    bat_dict = data_dict

    exog_bat_id = 'b2c4'
    split_parameter = 0.5 #currently not used

    params = None

    mape_list, rmse_list, cycle_life_error_list, cycle_life_error_percentage_list = test_model(split_parameter, bat_dict, exog_bat_id, auto=False, plot=False)


    print("mape mean: ", np.mean(mape_list))
    print("mse mean: ", np.mean(rmse_list))
    print("cycle life prediction error mean: ", np.mean(cycle_life_error_list))
    print("cycle life prediction percentage error mean: ", np.mean(cycle_life_error_percentage_list)*100)

    print("mape variance: ", np.var(mape_list))
    print("mse variance: ", np.var(rmse_list))
    print("cycle life prediction variance: ", np.var(cycle_life_error_list))



