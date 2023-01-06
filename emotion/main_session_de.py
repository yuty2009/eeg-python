# -*- coding:utf-8 -*-

import os
import numpy as np
from common.linear import *
import seedreader


setname = 'seed'
fs = 200
n_classes = 3
chanset = np.arange(62)
n_channels = len(chanset)
datapath = 'E:/eegdata/emotion/seed/ExtractedFeatures/'
sessions = seedreader.get_session_list(datapath)

outpath = os.path.join(datapath, 'output')
if not os.path.exists(outpath):
    os.makedirs(outpath)

test_accus = np.zeros((len(sessions)))
for ss in range(len(sessions)):
    session = sessions[ss]
    print('Load EEG epochs for ' + session)
    data_sess, labels_sess = seedreader.get_session_feature(datapath, session, 'de_LDS')
    # the first 9 trials for training
    data_train = data_sess[:9]
    labels_train = labels_sess[:9]
    data_train = np.concatenate(data_train)
    features_train = np.reshape(data_train, [data_train.shape[0], -1])
    labels_train = np.concatenate(labels_train)
    # the last 6 trials for testing
    data_test = data_sess[9:]
    labels_test = labels_sess[9:]
    data_test = np.concatenate(data_test)
    features_test = np.reshape(data_test, [data_test.shape[0], -1])
    labels_test = np.concatenate(labels_test)

    model = SoftmaxClassifier()
    W = model.fit(features_train, labels_train)
    labels_predict, _ = model.predict(features_test)
    test_accus[ss] = np.mean(np.equal(labels_predict, labels_test).astype(np.float32))
    print("Session %s, Accuracy %.4f" % (session, test_accus[ss]))

print(f'Overall accuracy: {np.mean(test_accus): .3f}')

import matplotlib.pyplot as plt
x = np.arange(len(test_accus))
plt.bar(x, test_accus)
plt.title('Averaged accuracy for all subjects')
plt.xticks(x, sessions)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
