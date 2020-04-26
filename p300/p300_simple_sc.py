# -*- coding:utf-8 -*-

import numpy as np
from eegreader_cnt import *
from common.regressor import *

datapath = 'E:/eegdata/scutbci/p300speller/'

fs = 250
f2 = 20
order = 6
fb, fa = butter(order, 2 * f2 / fs, btype='low')
# show_filter(fb, fa, fs)

dfs = 6
sampleseg = [0, int(0.6 * fs)]
chanset = np.arange(30)

subject = 'p300speller_yutianyou_20170313_1'
file_train = datapath + subject + '.cnt'

print('Load and extract continuous EEG into epochs for train data')
data, target = load_eegdata(file_train, [fb, fa])
print('Extract P300 features from epochs for train data')
feature, label = extract_feature(data, target, sampleseg, chanset, dfs)
num_trials, num_repeats, num_chars, num_features = feature.shape

num_perms = 1
num_folds = 10
num_total = num_trials
num_test = int(np.floor(num_total/num_folds))
num_train = num_total - num_test
acc_train = np.zeros([num_perms, num_folds, num_repeats])
print('%d - %d fold cross-validation' % (num_perms, num_folds))
for perm in range(num_perms):
    print('perm %d' % perm)
    index_total = np.random.permutation(np.arange(num_total))
    for fold in range(num_folds):
        print('fold %d' % fold)
        index_test = index_total[fold*num_test:(fold+1)*num_test]
        index_train = np.setdiff1d(index_total, index_test)
        feature_train = feature[index_train]
        feature_test = feature[index_test]
        label_train = label[index_train]
        label_test = label[index_test]
        target_train = target[index_train]
        target_test = target[index_test]

        X_train = np.reshape(feature_train, [-1, num_features])
        y_train = np.reshape(label_train, [-1])

        w, b = ridgereg(y_train, X_train)

        X_test = np.reshape(feature_test, [-1, num_features])
        y_predict = np.dot(X_test, w) + b

        targetPredict = np.zeros([num_test, num_repeats])
        for trial in range(num_test):
            ytrial = y_predict[trial*num_repeats*num_chars:(trial+1)*num_repeats*num_chars]
            ytrial = np.reshape(ytrial, [num_repeats, num_chars])
            for repeat in range(num_repeats):
                yavg = np.mean(ytrial[0:repeat+1,:], axis=0)
                targetPredict[trial, repeat] = np.argmax(yavg) + 1

        for i in range(num_repeats):
            acc_train[perm, fold, i] = np.mean(np.array(targetPredict[:,i] == target_test).astype(int))
acc_mean = np.mean(np.mean(acc_train, axis=0), axis=0)

import matplotlib.pyplot as plt
plt.plot(np.arange(num_repeats)+1, acc_mean*100, 'k-')
plt.title('Character Recognition Rate for ' + subject)
plt.xlabel('Repeat [n]')
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()