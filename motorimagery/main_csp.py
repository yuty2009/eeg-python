# -*- coding:utf-8 -*-

import numpy as np
from common.linear import *
from motorimagery.eegreader import *


datapath = 'E:/bcicompetition/bci2005/IVa/'
subjects = ['aa', 'al', 'av', 'aw', 'ay']

fs = 100
f1 = 7
f2 = 30
order = 6
fb, fa = signal.butter(order, [2*f1/fs, 2*f2/fs], btype='band')
show_filter(fb, fa, fs)

sampleseg = [50, 350]
chanset = np.arange(118)

accTest = np.zeros(len(subjects))
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_dataset(datapath, subject)
    print('Extract CSP features from epochs for subject ' + subject)
    RsTrain, labelTrain = extract_variance(dataTrain, targetTrain, [fb, fa], sampleseg, chanset)
    RsTest, labelTest = extract_variance(dataTest, targetTest, [fb, fa], sampleseg, chanset)
    num_train, num_channels = RsTrain.shape[0:2]
    num_test = RsTest.shape[0]

    num_features = 6
    WCSP = trainCSP2(RsTrain, labelTrain, num_features)

    X_train = np.zeros([num_train, num_features])
    for i in range(num_train):
        temp = np.abs(np.diag(np.dot(np.dot(WCSP.T, RsTrain[i]), WCSP)))
        temp = np.log(temp / np.sum(temp))
        X_train[i, :] = temp
    y_train = labelTrain
    y_train[np.argwhere(y_train == 2)] = -1

    model = LogisticRegression()
    w, b = model.fit(X_train, y_train)

    X_test = np.zeros([num_test, num_features])
    for i in range(num_test):
        temp = np.abs(np.diag(np.dot(np.dot(WCSP.T, RsTest[i]), WCSP)))
        temp = np.log(temp / np.sum(temp))
        X_test[i, :] = temp
    y_test = labelTest
    y_test[np.argwhere(y_test == 2)] = -1

    y_predict = model.predict(X_test)

    accTest[ss] = np.mean(np.array(np.sign(y_predict) == y_test).astype(int))

print(np.mean(accTest))

import matplotlib.pyplot as plt
x = np.arange(len(accTest))
plt.bar(x, accTest*100)
plt.title('Accuracy for the five subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
