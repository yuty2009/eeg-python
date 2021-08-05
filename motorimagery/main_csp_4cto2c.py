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


f1 = 7
f2 = 30
order = 3
fb, fa = signal.butter(order, [2*f1/fs, 2*f2/fs], btype='bandpass')
# fb, fa = signal.cheby2(order, 40, [2*f1/fs, 2*f2/fs], btype='bandpass')
# show_filter(fb, fa, fs)

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
        RsTrain, labelTrain = extract_variance(dataTrain1, targetTrain1, [fb, fa], sampleseg, chanset)
        RsTest, labelTest = extract_variance(dataTest1, targetTest1, [fb, fa], sampleseg, chanset)
        num_train, num_channels = RsTrain.shape[0:2]
        num_test = RsTest.shape[0]

        WCSP = trainCSP2(RsTrain, labelTrain, num_filters)

        X_train = np.zeros([num_train, num_filters])
        for i in range(num_train):
            temp = np.abs(np.diag(np.dot(np.dot(WCSP.T, RsTrain[i]), WCSP)))
            temp = np.log(temp / np.sum(temp))
            X_train[i, :] = temp
        y_train = labelTrain
        y_train[np.argwhere(y_train == labels[0])] = 1
        y_train[np.argwhere(y_train == labels[1])] = -1

        model = svm.SVC(kernel='linear')
        model.fit(X_train, y_train)
        # model = LogisticRegression()
        # w, b = model.fit(X_train, y_train)

        X_test = np.zeros([num_test, num_filters])
        for i in range(num_test):
            temp = np.abs(np.diag(np.dot(np.dot(WCSP.T, RsTest[i]), WCSP)))
            temp = np.log(temp / np.sum(temp))
            X_test[i, :] = temp
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

