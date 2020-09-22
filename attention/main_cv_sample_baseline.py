# -*- coding:utf-8 -*-

import os
import numpy as np
from sklearn.preprocessing import scale
import attention.eegreader_xml as ds
from common.linear import *


datapath = 'e:/eegdata/attention/Headband/data4class/'
modelpath = os.path.join(datapath, 'models/')
data = ds.load_dataset(datapath)
labels_all = []
features_all = []
for data_sub in data:
    labels_all.append(data_sub['labels'])
    features_all.append((data_sub['features']))
labels_all = np.concatenate(labels_all, axis=0)
features_all = np.concatenate(features_all, axis=0)
dataset = ds.Dataset(features_all, labels_all)

num_folds = 10
dataset.set_num_folds(num_folds)
accs = np.zeros(num_folds)
for fold in range(num_folds):
    features_train, labels_train, features_test, labels_test = dataset.next_fold()
    features_train = scale(features_train)
    features_test = scale(features_test)
    model = SoftmaxClassifier()
    W = model.fit(features_train, labels_train)
    labels_predict, _ = model.predict(features_test)
    accs[fold] = np.mean(np.equal(labels_predict, labels_test).astype(np.float32))
    print("Fold %d, Accuracy %.4f" % (fold, accs[fold]))
print("Mean accuracy %.4f" % np.mean(accs))