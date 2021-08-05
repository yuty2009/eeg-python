# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from common.linear import *
from common.spatialfilter import *
from motorimagery.mireader import *


setname = 'bcicomp2005IVa'
fs = 100
n_classes = 2
chanset = np.arange(118)
# chanset = [
#     np.arange(13,22),
#     np.arange(32,39),
#     np.arange(49,58),
#     np.arange(67,76),
#     np.arange(86,95),
#     np.array([103, 105, 107, 111, 112, 113])
#     ]
# chanset = np.hstack(chanset)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2005/IVa/'
subjects = ['aa', 'al', 'av', 'aw', 'ay']

fstart = 7
fstop = 30
fstep = 4
f1s = np.arange(fstart, fstop, fstep)
f2s = np.arange(fstart+fstep, fstop+fstep, fstep)
bands = np.hstack((f1s[:,None], f2s[:,None]))
# bands = [[7, 30], [30, 40]]

timewin = [1.0, 4.0] # 0.5s pre-task data
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

accTest = np.zeros(len(subjects))
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    print('Extract CSP features from epochs for subject ' + subject)
    RsTrain, labelTrain = extract_variance_multiband(dataTrain, targetTrain, bands, sampleseg, chanset)
    RsTest, labelTest = extract_variance_multiband(dataTest, targetTest, bands, sampleseg, chanset)
    num_bands = len(RsTrain)
    num_train, num_channels = RsTrain[0].shape[0:2]
    num_test = RsTest[0].shape[0]

    WCSPs = []
    num_features = 6
    for k in range(num_bands):
        WCSP = trainCSP2(RsTrain[k], labelTrain, num_features)
        WCSPs.append(WCSP)
    
    X_train = np.zeros([num_train, num_bands, num_features])
    for k in range(num_bands):
        WCSP = WCSPs[k]
        for i in range(num_train):
            temp = np.abs(np.diag(np.dot(np.dot(WCSP.T, RsTrain[k][i]), WCSP)))
            temp = np.log(temp / np.sum(temp))
            X_train[i, k, :] = temp
    X_train = np.reshape(X_train, [num_train, -1])
    y_train = labelTrain
    y_train[np.argwhere(y_train == 2)] = -1

    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    # model = LogisticRegression()
    # model.fit(X_train, y_train)

    X_test = np.zeros([num_test, num_bands, num_features])
    for k in range(num_bands):
        WCSP = WCSPs[k]
        for i in range(num_test):
            temp = np.abs(np.diag(np.dot(np.dot(WCSP.T, RsTest[k][i]), WCSP)))
            temp = np.log(temp / np.sum(temp))
            X_test[i, k, :] = temp
    X_test = np.reshape(X_test, [num_test, -1])
    y_test = labelTest
    y_test[np.argwhere(y_test == 2)] = -1

    y_predict = model.predict(X_test)

    accTest[ss] = np.mean(np.array(np.sign(y_predict) == y_test).astype(int))

print(np.mean(accTest))

import matplotlib.pyplot as plt
x = np.arange(len(accTest))
plt.bar(x, accTest*100)
plt.title('Accuracy for the five subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
