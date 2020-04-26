# -*- coding:utf-8 -*-

import os
import keras
import numpy as np
import cnnmodel as cnn
import eegreader as ds

# 6 by 6  matrix
matrix='ABCDEF'+'GHIJKL'+'MNOPQR'+'STUVWX'+'YZ1234'+'56789_'

datapath = 'E:/bcicompetition/bci2005/II/'
modelpath = os.path.join(datapath, 'models/')

subject = 'Subject_A'
featureTrain, labelTrain, targetTrain, featureTest, labelTest, targetTest = ds.load_dataset(datapath, subject)
num_train, num_repeats, num_chars, num_samples, num_channels = featureTrain.shape
num_test = featureTest.shape[0]

X_train = np.reshape(featureTrain, [-1, num_samples, num_channels, 1])
y_train = np.reshape(labelTrain, [-1, 1])

X_test = np.reshape(featureTest, [-1, num_samples, num_channels, 1])

model = cnn.CNNModel()
optimizer = keras.optimizers.adam(lr=1e-3, decay=1e-6)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
history = model.fit(X_train, y_train, batch_size=50, epochs=10, validation_split = 0.2, verbose=1)
model.summary()

y_predict = model.predict(X_test)

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