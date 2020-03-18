# -*- coding:utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import cnnmodel as cnn
import eegreader as ds
from common.regressor import *

# 6 by 6  matrix
matrix='ABCDEF'+'GHIJKL'+'MNOPQR'+'STUVWX'+'YZ1234'+'56789_'

datapath = 'E:/bcicompetition/bci2005/II/'
modelpath = os.path.join(datapath, 'models/')

subject = 'Subject_A'
featureTrain, labelTrain, targetTrain, featureTest, labelTest, targetTest = ds.load_dataset(datapath, subject)
num_train, num_chars, num_repeats, num_samples, num_channels = featureTrain.shape
num_test = featureTest.shape[0]

X_train = np.reshape(featureTrain, [-1, num_samples, num_channels, 1])
y_train = np.reshape(labelTrain, [-1, 1])

X_test = np.reshape(featureTest, [-1, num_samples, num_channels, 1])

train = ds.load_standardized_dataset(X_train, y_train)

ypred, X, ytrue, lr, dropout = cnn.createmodel(train.features.shape[1:])

mse = tf.losses.mean_squared_error(ytrue, ypred)
train_step = tf.train.AdamOptimizer(lr).minimize(mse)

maxstep = 10000
reportstep = 100
savestep = 2000
lr_start = 1e-3

global_step = 0
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(global_step, global_step+maxstep):
        batch_X, batch_y = train.next_batch(50)
        lr_now = lr_start * (1 + 1e-4 * i) ** (-0.75)
        train_step.run(feed_dict={X: batch_X, ytrue: batch_y, lr: lr_now, dropout: 0.5})
        if i % reportstep == 0:
            train_loss = mse.eval(feed_dict={
                X: batch_X, ytrue: batch_y, dropout: 1.0})
            print('Step=%d, lr=%.4f, loss=%.4f' % (i, lr_now, train_loss))
        if (i + 1) % savestep == 0:
            saver.save(sess, modelpath+'p300_model', global_step=i+1)
    y_predict = ypred.eval(feed_dict={X: X_test, dropout: 1.0})

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
print(accTest)

import matplotlib.pyplot as plt
plt.plot(np.arange(num_repeats)+1, accTest*100, 'k-')
plt.title('Character Recognition Rate for ' + subject)
plt.xlabel('Repeat [n]')
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()