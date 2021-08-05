# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from common.linear import *
from common.bayesian.linear import *
from common.spatialfilter import *
from common.riemann.riemann import *
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

num_filters = 6

accTest = np.zeros(len(subjects))
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    print('Extract CSP features from epochs for subject ' + subject)
    RsTrain, labelTrain = extract_variance(dataTrain, targetTrain, [fb, fa], sampleseg, chanset)
    RsTest, labelTest = extract_variance(dataTest, targetTest, [fb, fa], sampleseg, chanset)
    num_train, num_channels = RsTrain.shape[0:2]
    num_test = RsTest.shape[0]

    WCSP = trainCSP2(RsTrain, labelTrain, num_filters)

    RcspTrain = np.zeros((num_train, num_filters, num_filters))
    for i in range(num_train):
        RcspTrain[i, :, :] = np.dot(np.dot(WCSP.T, RsTrain[i]), WCSP)
    y_train = labelTrain
    index1 = np.argwhere(y_train == 1)
    index2 = np.argwhere(y_train == 2)
    C1 = riemannmean(RcspTrain[index1.flatten(), :, :])
    C2 = riemannmean(RcspTrain[index2.flatten(), :, :])

    RcspTest = np.zeros((num_test, num_filters, num_filters))
    for i in range(num_test):
        RcspTest[i, :, :] = np.dot(np.dot(WCSP.T, RsTest[i]), WCSP)
    y_test = labelTest
    y_pred = np.zeros_like(y_test)
    for i in range(num_test):
        dist1 = riemanndistance(C1, RcspTest[i, :, :])
        dist2 = riemanndistance(C2, RcspTest[i, :, :])
        if dist1 <= dist2:
            y_pred[i] = 1
        else:
            y_pred[i] = 2

    accTest[ss] = np.mean(np.array(y_pred == y_test).astype(int))

print(np.mean(accTest))

import matplotlib.pyplot as plt
x = np.arange(len(accTest))
plt.bar(x, accTest*100)
plt.title('Accuracy for the five subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
