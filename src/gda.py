import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def create_cycle_life_pickle_file():
    '''only used for file formatting'''
    batch1 = load_data('/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/batch1.pkl')
    batch2 = load_data('/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/batch2.pkl')
    batch3 = load_data('/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/batch3.pkl')


    def write_file(data_dict, filename):
        """Write to pickle file format"""
        with open(os.path.join('data', filename), 'wb') as f:
            pickle.dump(data_dict, f)

    cycle_lifes = dict()

    for key in batch1.keys():
        cycle_lifes[key] = batch1[key]['cycle_life']

    for key in batch2.keys():
        cycle_lifes[key] = batch2[key]['cycle_life']

    for key in batch3.keys():
        cycle_lifes[key] = batch3[key]['cycle_life']

    write_file(cycle_lifes, 'cycle_lifes.pkl')


def manipulate_cycle_lifes(train1, test1, cycle_lifes):
    '''data manipulation and binning of the cycle lifes'''
    train = train1.values
    test = test1.values
    y = np.array([])
    y_test = np.array([])

    i = 0
    for row in train:
        a = row[1]
        i += 1
        #changing bat_id to cycle life
        if i % 100 == 0:
            y = np.append(y, cycle_lifes[a])

    y = np.digitize(y, [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])
    #y = np.digitize(y,[100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050,
    #                 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850,
    #                 1900])

    #y = np.digitize(y, [200, 400, 600, 800, 1000, 1200, 1400, 1600])

    i = 0
    for row in test:
        a = row[1]
        i += 1
        #changing bat_id to cycle life
        if i % 100 == 0:
            y_test = np.append(y_test, cycle_lifes[a])

    print(y_test)
    plt.plot(y_test)
    plt.show()
    y_test = np.digitize(y_test, [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])
    #y_test = np.digitize(y_test,
    #                [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050,
    #                 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850,
    #                 1900])
    #y_test = np.digitize(y_test, [200, 400, 600, 800, 1000, 1200, 1400, 1600])

    return y, y_test


def create_data(train_path, test_path, cycle_life_path):
    '''extract train and test sets from dataset and return these'''
    cycle_lifes = load_data(cycle_life_path)
    train1 = pd.read_csv(train_path)
    test1 = pd.read_csv(test_path)

    y, y_test = manipulate_cycle_lifes(train1, test1, cycle_lifes)

    train = train1.drop('bat_id', axis=1)
    test = test1.drop('bat_id', axis=1)

    train = train.values
    test = test.values

    i = 0

    X = []
    battery_data_train = np.array([])

    for row in train:
        a = row
        i += 1
        battery_data_train = np.append(battery_data_train, np.array(np.split(row, 26)))
        if i % 100 == 0:
            X.append(battery_data_train)
            battery_data_train = np.array([])

    X = np.array(X)
    X = np.delete(X, slice(0, 8), axis=1)

    battery_data_test = np.array([])
    i = 0
    X_test = []
    for row in test:
        i += 1
        battery_data_test = np.append(battery_data_test, np.array(np.split(row, 26)))
        if i % 100 == 0:
            X_test.append(battery_data_test)
            battery_data_test = np.array([])

    X_test = np.array(X_test)
    X_test = np.delete(X_test, slice(0, 8), axis=1)

    return X, y, X_test, y_test


def model(X, y, X_test):
    '''runs GDA model'''
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)

    predictions = clf.predict(X_test)
    return predictions


def evaluate(predictions, y_test):
    '''evaluates predictions'''
    i = 0
    for item in predictions:
        predictions[i] = (item+1)*100
        i += 1

    i = 0
    for item in y_test:
        y_test[i] = (item+1)*100
        i += 1

    print(y_test)
    print(predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(rmse)
    mape = mean_absolute_percentage_error(y_test, predictions)
    print(mape)

    plt.plot(predictions, label = "predictions")
    plt.plot(y_test, label = "true")
    plt.ylabel('cycle life')
    plt.xlabel('battery number')
    plt.legend()
    plt.show()

    print(len(predictions))
    print(len(y_test))
    plt.scatter(y_test, predictions, color='red', alpha=0.5)
    plt.plot([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
             [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])
    plt.ylabel('predictions')
    plt.xlabel('true')
    plt.show()

if __name__ == '__main__':
    #change the paths to your paths
    cycle_life_path = '/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/cycle_lifes.pkl'
    train_path = '/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/train.csv'
    test_path = '/Users/Hampus Carlens/Desktop/CS229/lib-capacity-fade/data/test.csv'
    X, y, X_test, y_test = create_data(train_path, test_path, cycle_life_path)

    predictions = model(X, y, X_test)
    evaluate(predictions, y_test)


