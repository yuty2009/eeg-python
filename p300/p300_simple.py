# -*- coding:utf-8 -*-

import numpy as np
import eegreader as ds
from common.regressor import *

# 6 by 6  matrixA
matrix='ABCDEF'+'GHIJKL'+'MNOPQR'+'STUVWX'+'YZ1234'+'56789_'

datapath = 'E:/bcicompetition/bci2005/II/'

subject = 'Subject_A'
featureTrain, labelTrain, targetTrain, featureTest, labelTest, targetTest = ds.load_dataset(datapath, subject)
num_train, num_repeats, num_chars, num_samples, num_channels = featureTrain.shape
num_test = featureTest.shape[0]

X_train = np.reshape(featureTrain, [-1, num_samples*num_channels])
y_train = np.reshape(labelTrain, [-1])

X_test = np.reshape(featureTest, [-1, num_samples*num_channels])

w, b = ridgereg(y_train, X_train)

y_predict = np.dot(X_test, w) + b

targetPredict = np.zeros([num_test, num_repeats], dtype=np.str)
for trial in range(num_test):
    ytrial = y_predict[trial*num_repeats*num_chars:(trial+1)*num_repeats*num_chars]
    ytrial = np.reshape(ytrial, [num_repeats, num_chars])
    for repeat in range(num_repeats):
        yavg = np.mean(ytrial[0:repeat+1,:], axis=0)
        row = np.argmax(yavg[6:])
        col = np.argmax(yavg[0:6])
        targetPredict[trial, repeat] = matrix[int(row*6+col)]

accTest = np.zeros(num_repeats)
for i in range(num_repeats):
    accTest[i] = np.mean(np.array(targetPredict[:,i] == targetTest).astype(int))
print(accTest)

import matplotlib.pyplot as plt
plt.plot(np.arange(num_repeats)+1, accTest*100, 'k-')
plt.title('Character Recognition Rate for ' + subject)
plt.xlabel('Repeat [n]')
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()