# -*- coding:utf-8 -*-

import itertools
import numpy as np
from sklearn import svm
from common.linear import *
from common.spatialfilter import *
from motorimagery.mireader import *


"""
setname = 'bcicomp2005IIIa'
fs = 250
n_classes = 4
chanset = np.arange(60)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2005/IIIa/'
subjects = ['k3', 'k6', 'l1']
"""

setname = 'bcicomp2008IIa'
fs = 250
n_classes = 4
chanset = np.arange(22)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2008/IIa/'
# datapath = '/Users/yuty2009/data/bcicompetition/bci2008/IIa/'
subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']


fstart = 7
fstop = 30
fstep = 4
f1s = np.arange(fstart, fstop, fstep)
f2s = np.arange(fstart+fstep, fstop+fstep, fstep)
# bands = np.hstack((f1s[:,None], f2s[:,None]))
bands = [[7, 30], [30, 40]]

timewin = [1.0, 4.0] # 0.5s pre-task data
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

num_filters = 6

num_classes = 4
tasks = list(itertools.combinations(np.arange(1, num_classes+1), 2))
num_tasks = len(tasks)

accTest = np.zeros((len(subjects), num_tasks))
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    print('Extract CSP features from epochs for subject ' + subject)
    for tt in range(len(tasks)):
        labels = tasks[tt]
        indexTrain1 = np.argwhere((targetTrain == labels[0]) | (targetTrain == labels[1]))
        indexTest1 = np.argwhere((targetTest == labels[0]) | (targetTest == labels[1]))
        dataTrain1 = dataTrain[indexTrain1.ravel(),:,:]
        dataTest1 = dataTest[indexTest1.ravel(),:,:]
        targetTrain1 = targetTrain[indexTrain1.ravel()]
        targetTest1 = targetTest[indexTest1.ravel()]
        RsTrain, labelTrain = extract_variance_multiband(dataTrain1, targetTrain1, bands, sampleseg, chanset)
        RsTest, labelTest = extract_variance_multiband(dataTest1, targetTest1, bands, sampleseg, chanset)
        num_bands = len(RsTrain)
        num_train, num_channels = RsTrain[0].shape[0:2]
        num_test = RsTest[0].shape[0]

        WCSPs = []
        for k in range(num_bands):
            WCSP = trainCSP2(RsTrain[k], labelTrain, num_filters)
            WCSPs.append(WCSP)
        
        X_train = np.zeros([num_train, num_bands, num_filters])
        for k in range(num_bands):
            WCSP = WCSPs[k]
            for i in range(num_train):
                temp = np.abs(np.diag(np.dot(np.dot(WCSP.T, RsTrain[k][i]), WCSP)))
                temp = np.log(temp / np.sum(temp))
                X_train[i, k, :] = temp
        X_train = np.reshape(X_train, [num_train, -1])
        y_train = labelTrain
        y_train[np.argwhere(y_train == labels[0])] = 1
        y_train[np.argwhere(y_train == labels[1])] = -1

        model = svm.SVC(kernel='linear')
        model.fit(X_train, y_train)
        # model = LogisticRegression()
        # model.fit(X_train, y_train)

        X_test = np.zeros([num_test, num_bands, num_filters])
        for k in range(num_bands):
            WCSP = WCSPs[k]
            for i in range(num_test):
                temp = np.abs(np.diag(np.dot(np.dot(WCSP.T, RsTest[k][i]), WCSP)))
                temp = np.log(temp / np.sum(temp))
                X_test[i, k, :] = temp
        X_test = np.reshape(X_test, [num_test, -1])
        y_test = labelTest
        y_test[np.argwhere(y_test == labels[0])] = 1
        y_test[np.argwhere(y_test == labels[1])] = -1

        y_predict = model.predict(X_test)

        accTest[ss, tt] = np.mean(np.array(np.sign(y_predict) == y_test).astype(int))

print(np.mean(accTest))

import matplotlib.pyplot as plt
x = np.arange(len(accTest))
plt.bar(x, np.mean(accTest, 1)*100)
plt.title('Accuracy for the five subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
