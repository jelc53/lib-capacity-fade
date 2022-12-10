import numpy as np
import pandas as pd
import pickle


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

data_dict = load_data('/Users/knataraj/Documents/Academic/Stanford/CS229/project/processed_data.pkl')
bat_dict = data_dict
print(bat_dict.keys())

numBat1 = 41
numBat2 = 43
numBat3 = 40
numBat = numBat1+numBat2+numBat3

test_ind = np.hstack((np.arange(0,(numBat1+numBat2),2),83))
train_ind = np.arange(1,(numBat1+numBat2-1),2)
valid_ind = np.arange(numBat-numBat3,numBat);
test_keys = []
train_keys = []
valid_keys = []
train_ind = np.hstack((train_ind, valid_ind))
test_data = []
train_data = []

count = 0
for i in bat_dict.keys():
    if count in test_ind:
        test_data.append(bat_dict[i])
        test_keys.append(i)
    else:
        train_data.append(bat_dict[i])
        train_keys.append(i)
    count += 1

def compute_variance1(ind):
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

    hundredind2 = find_nearest(Vcyc100, 2)
    tenind2 = find_nearest(Vcyc10, 2)
    a = np.log(np.abs(dqcyc100[hundredind2] - dqcyc10[tenind2]))

    return (diff, a)


#training features
train = np.zeros((100*len(train_ind), 27))
train = pd.DataFrame(train)
for i in range(len(train_ind)):
    j=0
    train.iloc[i*100:(i+1)*100,j]=train_keys[i]
    j+=1
    train.iloc[i*100:(i+1)*100,j]=train_data[i]['summary']['IR'][0:100]
    j+=1
    train.iloc[i*100:(i+1)*100, j]=train_data[i]['summary']['Tavg'][0:100]
    j+=1
    train.iloc[i*100:(i+1)*100, j]=train_data[i]['summary']['Tmin'][0:100]
    j+=1
    train.iloc[i*100:(i+1)*100, j]=train_data[i]['summary']['Tmax'][0:100]
    j+=1
    train.iloc[i*100:(i+1)*100, j]=train_data[i]['summary']['QC'][0:100]
    j+=1
    train.iloc[i*100:(i+1)*100, j]=train_data[i]['summary']['QD'][0:100]
    j+=1
    train.iloc[i*100:(i+1)*100, j]=train_data[i]['summary']['chargetime'][0:100]
    j+=1

    vec = compute_variance1(i)
    diff = vec[0]
    train.iloc[i*100:(i+1)*100, j] = np.log(np.var(diff))
    j+=1
    train.iloc[i*100:(i+1)*100, j] = np.log(np.abs(np.min(diff)))
    j+=1
    train.iloc[i*100:(i+1)*100, j] =np.log(np.abs(ss.skew(diff)))
    j+=1
    train.iloc[i*100:(i+1)*100, j] =np.log(np.abs(ss.kurtosis(diff)))
    j+=1
    train.iloc[i*100:(i+1)*100, j] =vec[1]
    j+=1

    x=np.arange(2,101)
    y = train_data[i]['summary']['QD'][1:100]
    slope2100 = np.corrcoef(x, y)[0][1]*np.std(y)/np.std(x)
    train.iloc[i*100:(i+1)*100, j] = slope2100
    j+=1
    train.iloc[i*100:(i+1)*100, j] = np.mean(y) - slope2100*np.mean(x)
    j+=1

    x=np.arange(91,101)
    y = train_data[i]['summary']['QD'][90:100]
    slope91100 = np.corrcoef(x, y)[0][1]*np.std(y)/np.std(x)
    train.iloc[i*100:(i+1)*100, j] = slope91100
    j+=1
    train.iloc[i*100:(i+1)*100, j] = np.mean(y) - slope91100*np.mean(x)
    j+=1
    train.iloc[i*100:(i+1)*100, j] = train_data[i]['summary']['QD'][1]
    j+=1
    train.iloc[i*100:(i+1)*100, j] = np.max(train_data[i]['summary']['QD'][:]) - train_data[i]['summary']['QD'][1]
    j+=1
    train.iloc[i*100:(i+1)*100, j] = train_data[i]['summary']['QD'][99]
    j+=1
    train.iloc[i*100:(i+1)*100, j] = np.mean(train_data[i]['summary']['chargetime'][1:6])
    j+=1
    train.iloc[i*100:(i+1)*100, j] = np.sum(train_data[i]['summary']['Tavg'][1:100])
    j+=1
    train.iloc[i*100:(i+1)*100, j] = np.min(train_data[i]['summary']['Tmin'][1:100])
    j+=1
    train.iloc[i*100:(i+1)*100, j] = np.max(train_data[i]['summary']['Tmax'][1:100])
    j+=1
    train.iloc[i*100:(i+1)*100, j] = train_data[i]['summary']['IR'][1]
    j+=1
    train.iloc[i*100:(i+1)*100, j] = train_data[i]['summary']['IR'][99] - train_data[i]['summary']['IR'][1]
    j+=1
    #print(j)
    train.iloc[i*100:(i+1)*100, j] = np.min(train_data[i]['summary']['IR'][1:100])

train = pd.DataFrame(train)
train.columns=['bat_id', 'IR', 'Tavg', 'Tmin', 'Tmax', 'QC', 'QD', 'chargetime', 'Variance', 'log|min(delta(Q(V)))|','Skew',
               'Kurtosis', 'log|delta(Q(V=2))|', 'slope 2-100', 'intercept 2-100',
               'slope 91-100', 'intercept 91-100', 'QD cyc 2', 'maxQd - Qdcyc2', 'Qd cyc 100',
               'chargetimeavg cyc1-5', 'Temp integral', 'min temp', 'max temp', 'IR cyc 2', 'IR cyc100-cyc2',
               'min IR']
train
train.to_csv('/Users/knataraj/Documents/Academic/Stanford/CS229/project/train.csv')

def compute_variance2(ind):
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

    hundredind2 = find_nearest(Vcyc100, 2)
    tenind2 = find_nearest(Vcyc10, 2)
    a = np.log(np.abs(dqcyc100[hundredind2] - dqcyc10[tenind2]))
    return (diff, a)


#test features
test = np.zeros((100*len(test_ind), 27))
test = pd.DataFrame(test)
for i in range(len(test_ind)):
    j=0
    test.iloc[i*100:(i+1)*100,j]=test_keys[i]
    j+=1
    test.iloc[i*100:(i+1)*100,j]=test_data[i]['summary']['IR'][0:100]
    j+=1
    test.iloc[i*100:(i+1)*100, j]=test_data[i]['summary']['Tavg'][0:100]
    j+=1
    test.iloc[i*100:(i+1)*100, j]=test_data[i]['summary']['Tmin'][0:100]
    j+=1
    test.iloc[i*100:(i+1)*100, j]=test_data[i]['summary']['Tmax'][0:100]
    j+=1
    test.iloc[i*100:(i+1)*100, j]=test_data[i]['summary']['QC'][0:100]
    j+=1
    test.iloc[i*100:(i+1)*100, j]=test_data[i]['summary']['QD'][0:100]
    j+=1
    test.iloc[i*100:(i+1)*100, j]=test_data[i]['summary']['chargetime'][0:100]
    j+=1

    vec = compute_variance2(i)
    diff = vec[0]
    test.iloc[i*100:(i+1)*100, j] = np.log(np.var(diff))
    j+=1
    test.iloc[i*100:(i+1)*100, j] = np.log(np.abs(np.min(diff)))
    j+=1
    test.iloc[i*100:(i+1)*100, j] =np.log(np.abs(ss.skew(diff)))
    j+=1
    test.iloc[i*100:(i+1)*100, j] =np.log(np.abs(ss.kurtosis(diff)))
    j+=1
    test.iloc[i*100:(i+1)*100, j] =vec[1]
    j+=1

    x=np.arange(2,101)
    y = test_data[i]['summary']['QD'][1:100]
    slope2100 = np.corrcoef(x, y)[0][1]*np.std(y)/np.std(x)
    test.iloc[i*100:(i+1)*100, j] = slope2100
    j+=1
    test.iloc[i*100:(i+1)*100, j] = np.mean(y) - slope2100*np.mean(x)
    j+=1

    x=np.arange(91,101)
    y = test_data[i]['summary']['QD'][90:100]
    slope91100 = np.corrcoef(x, y)[0][1]*np.std(y)/np.std(x)
    test.iloc[i*100:(i+1)*100, j] = slope91100
    j+=1
    test.iloc[i*100:(i+1)*100, j] = np.mean(y) - slope91100*np.mean(x)
    j+=1
    test.iloc[i*100:(i+1)*100, j] = test_data[i]['summary']['QD'][1]
    j+=1
    test.iloc[i*100:(i+1)*100, j] = np.max(test_data[i]['summary']['QD'][:]) - test_data[i]['summary']['QD'][1]
    j+=1
    test.iloc[i*100:(i+1)*100, j] = test_data[i]['summary']['QD'][99]
    j+=1
    test.iloc[i*100:(i+1)*100, j] = np.mean(test_data[i]['summary']['chargetime'][1:6])
    j+=1
    test.iloc[i*100:(i+1)*100, j] = np.sum(test_data[i]['summary']['Tavg'][1:100])
    j+=1
    test.iloc[i*100:(i+1)*100, j] = np.min(test_data[i]['summary']['Tmin'][1:100])
    j+=1
    test.iloc[i*100:(i+1)*100, j] = np.max(test_data[i]['summary']['Tmax'][1:100])
    j+=1
    test.iloc[i*100:(i+1)*100, j] = test_data[i]['summary']['IR'][1]
    j+=1
    test.iloc[i*100:(i+1)*100, j] = test_data[i]['summary']['IR'][99] - test_data[i]['summary']['IR'][1]
    j+=1
    #print(j)
    test.iloc[i*100:(i+1)*100, j] = np.min(test_data[i]['summary']['IR'][1:100])

test = pd.DataFrame(test)
test.columns=['bat_id', 'IR', 'Tavg', 'Tmin', 'Tmax', 'QC', 'QD', 'chargetime', 'Variance', 'log|min(delta(Q(V)))|',
              'Skew', 'Kurtosis', 'log|delta(Q(V=2))|', 'slope 2-100', 'intercept 2-100',
               'slope 91-100', 'intercept 91-100', 'QD cyc 2', 'maxQd - Qdcyc2', 'Qd cyc 100',
               'chargetimeavg cyc1-5', 'Temp integral', 'min temp', 'max temp', 'IR cyc 2', 'IR cyc100-cyc2',
               'min IR']
test
test.to_csv('/Users/knataraj/Documents/Academic/Stanford/CS229/project/test.csv')
