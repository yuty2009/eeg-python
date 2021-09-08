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

"""
setname = 'bcicomp2005IIIa'
fs = 250
n_classes = 4
chanset = np.arange(60)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2005/IIIa/'
subjects = ['k3', 'k6', 'l1']
"""
"""
setname = 'bcicomp2008IIa'
fs = 250
n_classes = 4
chanset = np.arange(22)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2008/IIa/'
# datapath = '/Users/yuty2009/data/bcicompetition/bci2008/IIa/'
subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
"""

order, f1, f2 = 3, 7, 30
fb, fa = signal.butter(order, [2*f1/fs, 2*f2/fs], btype='bandpass')
# fb, fa = signal.cheby2(order, 30, [2*f1/fs, 2*f2/fs], btype='bandpass')
# show_filter(fb, fa, fs)

timewin = [1.0, 4.0] # 0.5s pre-task data
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

n_filters = 3

train_accu = np.zeros(len(subjects))
test_accu = np.zeros(len(subjects))
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    print('Extract raw features from epochs for subject ' + subject)
    featTrain, labelTrain = extract_rawfeature(dataTrain, targetTrain, sampleseg, chanset, [fb, fa])
    featTest, labelTest = extract_rawfeature(dataTest, targetTest, sampleseg, chanset, [fb, fa])
    n_train = featTrain.shape[0]
    n_test = featTest.shape[0]

    csp = CSP(n_filters)
    wcsp = csp.fit(featTrain, labelTrain)
    x_train = np.zeros([n_train, 2*n_filters])
    for i in range(n_train):
        x_train[i] = csp.transform(featTrain[i], wcsp)
    y_train = labelTrain
    y_train[np.argwhere(y_train == 2)] = -1

    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)
    # model = LogisticRegression()
    # w, b = model.fit(X_train, y_train)

    y_pred_train = model.predict(x_train)
    train_accu[ss] = np.mean(np.array(y_pred_train == y_train).astype(int))

    x_test = np.zeros([n_test, 2*n_filters])
    for i in range(n_test):
        x_test[i] = csp.transform(featTest[i], wcsp)
    y_test = labelTest
    y_test[np.argwhere(y_test == 2)] = -1

    y_pred = model.predict(x_test)
    # y_pred, _ = model.predict(X_test)

    test_accu[ss] = np.mean(np.array(y_pred == y_test).astype(int))

    print(f'Subject {subject} train_accu: {train_accu[ss]: .3f}, test_accu: {test_accu[ss]: .3f}')

print(f'Overall accuracy: {np.mean(test_accu): .3f}')

import matplotlib.pyplot as plt
x = np.arange(len(test_accu))
plt.bar(x, test_accu*100)
plt.title('Averaged accuracy for all subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
