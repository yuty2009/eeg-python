# -*- coding:utf-8 -*-

import os
import numpy as np
from sklearn.preprocessing import scale
import emotion.deap.eegreader_dat as ds
from common.linear import *


datapath = 'e:/eegdata/emotion/deap/data_preprocessed_python/'
# datapath = '/home/yuty2009/data/eegdata/emotion/deap/data_preprocessed_python/'
modelpath = os.path.join(datapath, 'models/')
data = ds.load_dataset(datapath)

num_subjects = len(data)
num_tasks = data[0]['labels'].shape[1]
num_folds = 10
accs = np.zeros((num_subjects, num_tasks, num_folds))
for sub, data_sub in enumerate(data):
    for task in range(num_tasks):
        labels_task = data_sub['labels'][:, task]
        labels_task[labels_task == 0] = -1
        dataset = ds.Dataset(data_sub['features'], labels_task)
        dataset.set_num_folds(num_folds)
        for fold in range(num_folds):
            features_train, labels_train, features_test, labels_test = dataset.next_fold()
            features_train = scale(features_train)
            features_test = scale(features_test)
            model = LogisticRegression()
            # model = SoftmaxClassifier()
            W = model.fit(features_train, labels_train)
            labels_predict, _ = model.predict(features_test)
            accs[sub, task, fold] = np.mean(np.equal(labels_predict, labels_test).astype(np.float32))
            print("Subject %d, Task %d, Fold %d, Accuracy %.4f" % (sub, task, fold, accs[sub, task, fold]))
        print("Subject %d, Task %d, Mean accuracy %.4f" % (sub, task, np.mean(accs[sub, task])))
np.savetxt("cv_single_subject.csv", np.mean(accs, axis=2), delimiter=',')
print(np.mean(np.mean(accs, axis=2)))