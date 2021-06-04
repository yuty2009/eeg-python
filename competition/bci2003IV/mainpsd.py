# -*- coding:utf-8 -*-

import numpy as np
import sklearn.svm as svm
from common.datawrapper import *
from common.temporalfilter import *
import competition.bci2003IV.eegreader as ds


datapath = 'E:/bcicompetition/bci2003/IV/sp1s_aa_1000Hz.mat'
labelpath = 'E:/bcicompetition/bci2003/IV/labels_data_set_iv.txt'
x_train, y_train, x_test, y_test = ds.load_dataset(datapath, labelpath)
print('Data loaded with %d training samples and %d testing samples' % (y_train.size, y_test.size))

fs = 1000
bands = [0.5,4,8,14,30]
features_train = ds.extract_bandpower(x_train, bands, fs)
features_test = ds.extract_bandpower(x_test, bands, fs)

model = svm.SVC()
model.fit(features_train, y_train)

y_pred = model.predict(features_train)
acc = sum(y_pred == y_train) / len(y_pred)
print('Train Accuracy: %f' % (acc*100))
