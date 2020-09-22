# -*- coding:utf-8 -*-

import os
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
import attention.eegreader_xml as ds
from common.pytorch import *


# datapath = 'e:/eegdata/attention/Headband/data4class/'
datapath = '/home/yuty2009/data/eegdata/attention/Headband/data4class/'
modelpath = os.path.join(datapath, 'models/')
data = ds.load_dataset(datapath)
labels_all = []
features_all = []
for data_sub in data:
    labels_sub = data_sub['labels']
    # make it a binary classification problem
    labels_sub[np.argwhere(labels_sub!=1)] = 0
    labels_all.append(labels_sub)
    features_sub = data_sub['features']
    features_sub = np.reshape(features_sub, [features_sub.shape[0], -1])
    features_all.append(features_sub)

num_subjects = len(data)
accs = np.zeros(num_subjects)
for subject in range(num_subjects):
    labels_test = labels_all[subject]
    features_test = features_all[subject]
    # features_test = scale(features_test)
    labels_train = labels_all[0:subject] + labels_all[subject+1:]
    features_train = features_all[0:subject] + features_all[subject+1:]
    labels_train = np.concatenate(labels_train, axis=0)
    features_train = np.concatenate(features_train, axis=0)
    # features_train = scale(features_train)
    model = SoftmaxClassifier()
    W = model.fit(features_train, labels_train)
    labels_predict, _ = model.predict(features_test)
    # model = LogisticRegression()
    # w, b = model.fit(features_train, labels_train)
    # labels_predict, _ = model.predict(features_test)
    # model = LogisticRegression(random_state=0)
    # model.fit(features_train, labels_train)
    # labels_predict = model.predict(features_test)
    accs[subject] = np.mean(np.equal(labels_predict, labels_test).astype(np.float32))
    print("Leave-one-subject-out %d, Accuracy %.4f" % (subject, accs[subject]))
print("Mean accuracy %.4f" % np.mean(accs))