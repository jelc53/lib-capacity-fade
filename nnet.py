import keras
import pickle
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

###Data loading
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

#Change the path to where you've put the data
data_dict = load_data('/Users/knataraj/Documents/Academic/Stanford/CS229/project/processed_data.pkl')
bat_dict = data_dict
print(bat_dict.keys())

###Train/test split
numBat1 = 41
numBat2 = 43
numBat3 = 40
numBat = numBat1+numBat2+numBat3


test_ind = np.hstack((np.arange(0,(numBat1+numBat2),2),83))
train_ind = np.arange(1,(numBat1+numBat2-1),2)
valid_ind = np.arange(numBat-numBat3,numBat);

train_ind = np.hstack((train_ind, valid_ind))
print(len(train_ind))

test_data = []
#cycle_life_valid_data = []
train_data = []

count = 0
for i in bat_dict.keys():
    if count in test_ind:
        test_data.append(bat_dict[i])
    #elif count in valid_ind:
     #   cycle_life_valid_data.append(bat_dict[i]['cycle_life'][0][0])
    else:
        train_data.append(bat_dict[i])
    count += 1

###Setting up the target variable
y = np.zeros(81)  #training y
yt= np.zeros(len(test_ind))  #testing y
for i in range(len(y)):
    y[i] = len(train_data[i]['summary']['cycle'])
for i in range(len(yt)):
    yt[i] = len(test_data[i]['summary']['cycle'])

print(y, yt)


###Setting up training, test matrices, first populating with first 100 cycles worth of
###internal resistance and average temperature (so each column represents data for one of these cycles)
train_mat = np.zeros((len(train_ind), 201))
test_mat = np.zeros((len(test_ind), 201))
print(train_mat.shape)
for i in range(len(train_ind)):
    train_mat[i,:200] = np.hstack((train_data[i]['summary']['IR'][0:100], train_data[i]['summary']['Tavg'][0:100]))

    print(i)

for i in range(len(test_ind)):
    test_mat[i,:200] = np.hstack((test_data[i]['summary']['IR'][0:100], test_data[i]['summary']['Tavg'][0:100]))

    print(i)

###Last feature, which is variance of dq_100(voltage) - dq_10(voltage)
#For the training matrix:
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def compute_variance(ind):
    Vcyc10 = train_data[ind]['cycles']['10']['V']
    maxi = np.argmax(Vcyc10)
    Vcyc10 = Vcyc10[maxi:]

    Vcyc100 = train_data[ind]['cycles']['100']['V']
    maxi1 = np.argmax(Vcyc100)
    Vcyc100 = Vcyc100[maxi1:]

    dqcyc10 = train_data[ind]['cycles']['10']['Qd']
    dqcyc10 = dqcyc10[maxi:]

    dqcyc100 = train_data[ind]['cycles']['100']['Qd']
    dqcyc100 = dqcyc100[maxi1:]

    minV = min(train_data[10]['cycles']['10']['V'])
    maxV = max(train_data[10]['cycles']['10']['V'])
    seq = np.arange(minV, maxV, .01)

    diff = np.zeros(len(seq))
    i = 0
    for elem in seq:
        hundredind = find_nearest(Vcyc100, elem)
        tenind = find_nearest(Vcyc10, elem)
        diff[i] = dqcyc100[hundredind] - dqcyc10[tenind]
        i+=1

    return np.log(np.var(diff))

for i in range(81):
    currvar = compute_variance(i)
    train_mat[i][200] = currvar

#For the test matrix
def compute_variance_t(ind):
    Vcyc10 = test_data[ind]['cycles']['10']['V']
    maxi = np.argmax(Vcyc10)
    Vcyc10 = Vcyc10[maxi:]

    Vcyc100 = test_data[ind]['cycles']['100']['V']
    maxi1 = np.argmax(Vcyc100)
    Vcyc100 = Vcyc100[maxi1:]

    dqcyc10 = test_data[ind]['cycles']['10']['Qd']
    dqcyc10 = dqcyc10[maxi:]

    dqcyc100 = test_data[ind]['cycles']['100']['Qd']
    dqcyc100 = dqcyc100[maxi1:]

    minV = min(test_data[10]['cycles']['10']['V'])
    maxV = max(test_data[10]['cycles']['10']['V'])
    seq = np.arange(minV, maxV, .01)
    diff = np.zeros(len(seq))

    i = 0
    for elem in seq:
        hundredind = find_nearest(Vcyc100, elem)
        tenind = find_nearest(Vcyc10, elem)
        diff[i] = dqcyc100[hundredind] - dqcyc10[tenind]
        i+=1

    return np.log(np.var(diff))


for i in range(len(test_ind)):
    currvar = compute_variance_t(i)
    test_mat[i][200] = currvar

###Modelling CODE
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
model = Sequential()
model.add(Dense(1000, input_dim=201, activation="relu"))
model.add(Dense(1))

#loss='mse',
 #   metrics=[tf.keras.metrics.MeanSquaredError()])
model.compile(loss="mse", optimizer="adam", metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(train_mat,y, epochs=5000, batch_size=100)
_, accuracy = model.evaluate(train_mat, y)
#print("Model accuracy: %.2f"% (accuracy*100))


###Predictions and MAPE calculation
#Training:
predictions = model.predict(train_mat)
y = y.reshape(-1,1)
print(np.mean(np.abs(predictions-y)/y))

#Test
predictionst = model.predict(test_mat)
yt = yt.reshape(-1,1)
print(np.mean(np.abs(predictionst-yt)/yt))
