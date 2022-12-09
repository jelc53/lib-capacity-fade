from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import six
import sys
sys.modules['sklearn.externals.six'] = six
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import pickle
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data_dict = load_data('/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/processed_data_lite.pkl')
bat_dict = data_dict


df = pd.DataFrame(np.array(bat_dict['b1c0']['summary']['QD']))

TEST_SIZE = 200
train, test = df.iloc[:-TEST_SIZE], df.iloc[-TEST_SIZE:]
x_train, x_test = np.array(range(train.shape[0])), np.array(range(train.shape[0], df.shape[0]))

#plot_acf(df, lags=100)

#plt.show()
'''
data_diff = df.diff().dropna()
data_diff.plot(figsize=(15,5))
#plt.show()

plot_acf(np.array(data_diff))
#plt.show()
'''



'''
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(train)
ax.plot(test)
plt.show()
'''
'''
train.plot()
test.plot()
plt.show()
'''

model = auto_arima(train, start_p=1, start_q=1,
                      test='adf',
                      max_p=5, max_q=5,
                      m=1,
                      d=1,
                      seasonal=False,
                      start_P=0,
                      D=None,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

prediction, confint = model.predict(n_periods=TEST_SIZE, return_conf_int=True)
cf = pd.DataFrame(confint)

prediction_series = pd.Series(prediction,index=test.index)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(df.values)
ax.plot(prediction_series)
ax.fill_between(prediction_series.index,
                cf[0],
                cf[1],color='grey',alpha=.3)

plt.show()

