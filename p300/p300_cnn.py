# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import p300.eegreader as ds
from p300.cnnmodel import *


# 6 by 6  matrix
matrix='ABCDEF'+'GHIJKL'+'MNOPQR'+'STUVWX'+'YZ1234'+'56789_'

datapath = 'E:/bcicompetition/bci2005/II/'
modelpath = os.path.join(datapath, 'models/')

subject = 'Subject_A'
featureTrain, labelTrain, targetTrain, featureTest, labelTest, targetTest = ds.load_dataset(datapath, subject)
num_train, num_repeats, num_chars, num_samples, num_channels = featureTrain.shape
num_test = featureTest.shape[0]

X_train = np.reshape(featureTrain, [-1, 1, num_samples, num_channels])
y_train = np.reshape(labelTrain, [-1, 1])
trainset = ds.load_standardized_dataset(X_train, y_train)

X_test = np.reshape(featureTest, [-1, 1, num_samples, num_channels])
y_test = np.reshape(labelTest, [-1, 1])

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

cnn = CNNModel(dropout=0.5).to(device)
optimizer = optim.Adam(cnn.parameters(), lr=1e-3)

# train
epochs = 20
batch_size = 50
epoch_steps = np.ceil(trainset.num_examples/batch_size).astype('int')
for epoch in range(epochs):
    for step in range(epoch_steps):
        X_batch, y_batch = trainset.next_batch(batch_size)
        X_batch = torch.tensor(X_batch, device=device).type(torch.float32)
        y_batch = torch.tensor(y_batch, device=device).type(torch.float32)

        yp_batch = cnn(X_batch)
        loss = F.mse_loss(yp_batch, y_batch, reduction='sum')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics every 100 steps
        if (step + 1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss {:.4f}"
                  .format(epoch + 1, epochs,
                          (step + 1) * batch_size, trainset.num_examples,
                          loss.item()))

with torch.no_grad():
    X_test = torch.tensor(X_test, device=device).type(torch.float32)
    y_pred = cnn(X_test)
    y_pred = y_pred.cpu().data.numpy()

    targetPredict = np.zeros([num_test, num_repeats], dtype=np.str)
    for trial in range(num_test):
        y_trial = y_pred[trial*num_repeats*num_chars:(trial+1)*num_repeats*num_chars]
        y_trial = np.reshape(y_trial, [num_repeats, num_chars])
        for repeat in range(num_repeats):
            y_avg = np.mean(y_trial[0:repeat+1,:], axis=0)
            row = np.argmax(y_avg[6:])
            col = np.argmax(y_avg[0:6])
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