# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from common.linear import *
from common.spatialfilter import *
from common import datawrapper as ds
from motorimagery.mireader import *


"""
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

order, fstart, fstop, fstep, = 3, 4, 40, 4
f1s = np.arange(fstart, fstop, fstep)
f2s = np.arange(fstart+fstep, fstop+fstep, fstep)
fbanks = np.hstack((f1s[:,None], f2s[:,None]))
# fbanks = [[7, 30], [30, 40]]
n_fbanks = len(fbanks)

timewin = [1.0, 4.0] # 0.5s pre-task data
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

n_filters = 3

n_times = 1
n_folds = 10
train_accus = np.zeros([len(subjects), n_times, n_folds])
test_accus = np.zeros([len(subjects), n_times, n_folds])
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    print('Extract multi-band features from epochs for subject ' + subject)
    featTrain_bands = []
    featTest_bands = []
    for k in range(n_fbanks):
        f1, f2 = fbanks[k]
        fb, fa = signal.butter(order, [2*f1/fs, 2*f2/fs], btype='bandpass')
        # fb, fa = signal.cheby2(order, 30, [2*f1/fs, 2*f2/fs], btype='bandpass')
        featTrain, labelTrain = extract_rawfeature(dataTrain, targetTrain, sampleseg, chanset, [fb, fa])
        featTest, labelTest = extract_rawfeature(dataTest, targetTest, sampleseg, chanset, [fb, fa])
        featTrain_bands.append(featTrain)
        featTest_bands.append(featTest)
    featTrain_bands = np.transpose(np.stack(featTrain_bands, axis=0), [1, 0, 2, 3])
    featTest_bands = np.transpose(np.stack(featTest_bands, axis=0), [1, 0, 2, 3])
    n_train = featTrain.shape[1]
    n_test = featTest.shape[1]

    for t in range(n_times):
        np.random.seed(t)
        ds_train = ds.Dataset(featTrain_bands, labelTrain)
        ds_train.set_num_folds(n_folds)
        for fold in range(n_folds):
            '''for N times x K fold CV'''
            feaTrain_fold, labelTrain_fold, featTest_fold, labelTest_fold = ds_train.next_fold()

            fbcsp = FBCSP(n_filters)
            wfbcsp = fbcsp.fit(feaTrain_fold, labelTrain_fold)

            x_train = fbcsp.transform(feaTrain_fold, wfbcsp)
            y_train = labelTrain_fold - 1 # 

            model = SoftmaxClassifier()
            model.fit(x_train, y_train)

            y_pred_train, _ = model.predict(x_train)
            train_accus[ss] = np.mean(np.array(y_pred_train == y_train).astype(int))

            x_test = fbcsp.transform(featTest_fold, wfbcsp)
            y_test = labelTest_fold - 1

            # y_pred = model.predict(x_test)
            y_pred, _ = model.predict(x_test)
            test_accus[ss, t, fold] = np.mean(np.array(y_pred == y_test).astype(int))

    train_accu_cv = np.mean(train_accus[ss])
    test_accu_cv = np.mean(test_accus[ss])
    print(f'Subject {subject} train_accu: {train_accu_cv: .3f}, test_accu: {test_accu_cv: .3f}')

print(f'Overall accuracy: {np.mean(test_accus): .3f}')

test_accu_ss = np.mean(np.mean(test_accus, axis=-1),axis=-1)

import matplotlib.pyplot as plt
x = np.arange(len(test_accu_ss))
plt.bar(x, test_accu_ss*100)
plt.title('Averaged accuracy for all subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
