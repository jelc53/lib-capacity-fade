import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def create_dataframe():
    '''create dataframe from file'''
    data_dict = load_data('/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/processed_data_lite.pkl')
    bat_dict = data_dict

    df = pd.DataFrame(np.array(bat_dict['b2c1']['summary']['QD']))
    return df


def create_test_split(df, split_parameter):
    '''creating a train/test split'''
    test_size = int(split_parameter*len(df.values))
    df1 = df[1:test_size]
    df2 = df[1:1500]
    return df1, df2, test_size

def model_simple(df, split_parameter):
    '''performing simple exponential smoothing'''
    df1, df2, test_size = create_test_split(df, split_parameter)
    fit1 = SimpleExpSmoothing(df1).fit(use_boxcox=True)
    df[1:].plot(style='--', color='red', legend=True, label="true values")

    forecast = fit1.forecast(1000)
    forecast.plot(style='--', color='green', legend=True, label="fit1")
    plt.show()

def model_holt(df, split_parameter):
    '''performing holt exponential smoothing'''
    df1, df2, test_size = create_test_split(df, split_parameter)
    fit1 = Holt(df1, exponential=True, damped_trend=True).fit(use_boxcox=True)
    df[1:].plot(style='--', color='red', legend=True, label="true values")
    plt.legend()
    forecast = fit1.forecast(1000)
    forecast.plot(style='--', color='green', legend=True, label="prediction")
    plt.legend()
    plt.show()


    a = forecast.values[:len(df.values)-test_size]
    b = df.values[test_size:]
    plt.plot(a)
    plt.plot(b)
    plt.show()
    mape = mean_absolute_percentage_error(a, b)
    cycle_life = len(df.values)
    cycle_life_pred = test_size
    Q_nom = df.values[5]

    i = 0
    for value in forecast.values:
        if value/Q_nom < 0.8:
            cycle_life_pred += i
            break
        i += 1

    cycle_life_pred_error = abs(cycle_life-cycle_life_pred)
    return mape, cycle_life_pred_error

def model_holt_wint(df, split_parameter):
    '''performing holt winters exponential smoothing'''

    df1, df2, test_size = create_test_split(df, split_parameter)
    fit1 = ExponentialSmoothing(df1, trend='mul', seasonal=None).fit(use_boxcox=True)
    fit2 = ExponentialSmoothing(df2, trend='mul', seasonal=None).fit(use_boxcox=True)

    df[1:].plot(style='--', color='red', legend=True, label="true values")

    forecast = fit1.forecast(1000)
    forecast.plot(style='--', color='green', legend=True, label="prediction")
    #fit2.forecast(1000).plot(style='--', color='blue', legend=True, label="fit2")

    plt.show()

    a = forecast.values[:len(df.values)-test_size]
    b = df.values[test_size:]
    plt.plot(a)
    plt.plot(b)
    plt.show()
    mape = mean_absolute_percentage_error(a, b)
    cycle_life = len(df.values)
    cycle_life_pred = test_size
    Q_nom = df.values[5]

    i = 0
    for value in forecast.values:
        if value/Q_nom < 0.8:
            cycle_life_pred += i
            break
        i += 1

    cycle_life_pred_error = abs(cycle_life-cycle_life_pred)
    return mape, cycle_life_pred_error

if __name__ == '__main__':
    df = create_dataframe()
    split_paramter = 0.5

    #comment or uncomment to select model
    #mape, cycle_life_pred_error = model_holt_wint(df, split_paramter)
    #model_simple(df, split_paramter)
    mape, cycle_life_pred_error = model_holt(df, split_paramter)

    print(mape)
    print(cycle_life_pred_error)


