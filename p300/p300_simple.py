# -*- coding:utf-8 -*-

import numpy as np
from eegreader import *
from common.regressor import *

# 6 by 6  matrix
matrix='ABCDEF'+'GHIJKL'+'MNOPQR'+'STUVWX'+'YZ1234'+'56789_'

datapath = 'E:/bcicompetition/bci2005/II/'

subject = 'Subject_A'
featureTrain, labelTrain, targetTrain, featureTest, labelTest, targetTest = load_dataset(datapath, subject)
num_train, num_chars, num_repeats, num_features = featureTrain.shape
num_test = featureTest.shape[0]

X_train = np.reshape(featureTrain, [-1, num_features])
y_train = np.reshape(labelTrain, [-1])

w, b = ridgereg(y_train, X_train)

X_test = np.reshape(featureTest, [-1, num_features])
y_predict = np.dot(X_test, w) + b

targetPredict = np.zeros([num_test, num_repeats], dtype=np.str)
for trial in range(num_test):
    ytrial = y_predict[trial*num_chars*num_repeats:(trial+1)*num_chars*num_repeats]
    ytrial = np.reshape(ytrial, [num_chars, num_repeats])
    for repeat in range(num_repeats):
        yavg = np.mean(ytrial[:,0:repeat+1], axis=1)
        row = np.argmax(yavg[6:])
        col = np.argmax(yavg[0:6])
        targetPredict[trial, repeat] = matrix[int(row*6+col)]

accTest = np.zeros(num_repeats)
for i in range(num_repeats):
    accTest[i] = np.mean(np.array(targetPredict[:,i] == targetTest).astype(int))

import matplotlib.pyplot as plt
plt.plot(np.arange(num_repeats)+1, accTest*100, 'k-')
plt.title('Character Recognition Rate for ' + subject)
plt.xlabel('Repeat [n]')
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()