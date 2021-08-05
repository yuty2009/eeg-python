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
    Cmean = riemannmean(RcspTrain)
    X_train = tangentspace(RcspTrain, Cmean)
    y_train = labelTrain
    y_train[np.argwhere(y_train == 2)] = -1

    # model = svm.SVC(kernel='linear')
    # model.fit(X_train, y_train)
    model = BayesARDLogisticRegression()
    w, b = model.fit(X_train, y_train)

    RcspTest = np.zeros((num_test, num_filters, num_filters))
    for i in range(num_test):
        RcspTest[i, :, :] = np.dot(np.dot(WCSP.T, RsTest[i]), WCSP)
    X_test = tangentspace(RcspTest, Cmean)
    y_test = labelTest
    y_test[np.argwhere(y_test == 2)] = -1
    
    # y_pred = model.predict(X_test)
    y_pred, _ = model.predict(X_test)

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
