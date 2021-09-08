# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from common.linear import *
from common.bayesian.linear import *
from common.spatialfilter import *
from common.riemann.riemann import *
from mireader import *


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

n_filters = 3

test_accus = np.zeros(len(subjects))
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

    covzTrain = np.zeros((n_train, 2*n_filters, 2*n_filters))
    for i in range(n_train):
        z_trial = csp.project(featTrain[i], wcsp)
        covzTrain[i, :, :] = np.matmul(np.transpose(z_trial), z_trial)
    y_train = labelTrain
    index1 = np.argwhere(y_train == 1)
    index2 = np.argwhere(y_train == 2)
    C1 = riemannmean(covzTrain[index1.flatten(), :, :])
    C2 = riemannmean(covzTrain[index2.flatten(), :, :])

    covzTest = np.zeros((n_test, 2*n_filters, 2*n_filters))
    for i in range(n_test):
        z_trial = csp.project(featTest[i], wcsp)
        covzTest[i, :, :] = np.matmul(np.transpose(z_trial), z_trial)
    y_test = labelTest
    y_pred = np.zeros_like(y_test)
    for i in range(n_test):
        dist1 = riemanndistance(C1, covzTest[i, :, :])
        dist2 = riemanndistance(C2, covzTest[i, :, :])
        if dist1 <= dist2:
            y_pred[i] = 1
        else:
            y_pred[i] = 2

    test_accus[ss] = np.mean(np.array(y_pred == y_test).astype(int))

print(f'Overall accuracy: {np.mean(test_accus): .3f}')

import matplotlib.pyplot as plt
x = np.arange(len(test_accus))
plt.bar(x, test_accus*100)
plt.title('Averaged accuracy for all subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
