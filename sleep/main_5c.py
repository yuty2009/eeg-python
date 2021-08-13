# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import random_split
from common.utils import *
from common.torchutils import train_epoch, evaluate, DEVICE
from sleepreader import *
from sleepnet import DeepSleepNet, TinySleepNet


datapath = 'e:/eegdata/sleep/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
# datapath = '/Users/yuty2009/data/eegdata/sleep/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
data, labels, subjects = load_dataset_preprocessed(datapath+'processed/', n_subjects=39)
print('Data for %d subjects has been loaded' % len(data))
n_subjects = len(data)
n_timepoints = data[0].shape[-2]
n_classes = 5

n_folds = 20
test_fraction = 1.0 / n_folds
n_test = math.ceil(n_subjects * test_fraction)
valid_fraction = 0.1
n_train = n_subjects - n_test
n_valid = math.ceil(n_train * valid_fraction)
n_train = n_train - n_valid

idx_train, idx_valid, idx_test = random_split(range(n_subjects), [n_train, n_valid, n_test])

torch.manual_seed(0)
tf_epoch = TransformEpoch()

n_seqlen = 15
trainset = [SeqEEGDataset(data[i], labels[i], n_seqlen, tf_epoch) for i in idx_train]
validset = [SeqEEGDataset(data[i], labels[i], n_seqlen, tf_epoch) for i in idx_valid]
testset = [SeqEEGDataset(data[i], labels[i], n_seqlen, tf_epoch) for i in idx_test]
print('%d train vs %d valid vs %d test' % (len(trainset), len(validset), len(testset)))

model = TinySleepNet(n_timepoints, n_seqlen, n_classes).to(DEVICE)
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
print(model)

n_epochs = 200
batch_size = 20
train_accu = np.zeros((n_epochs, n_train), dtype=float)
train_loss = np.zeros((n_epochs, n_train), dtype=float)
valid_accu = np.zeros((n_epochs, n_valid), dtype=float)
valid_loss = np.zeros((n_epochs, n_valid), dtype=float)
test_accu = np.zeros((n_epochs, n_test), dtype=float)
test_loss = np.zeros((n_epochs, n_test), dtype=float)
for epoch in range(n_epochs):
    start = time.time()
    for sub in range(n_train):
        train_accu_sub, train_loss_sub = train_epoch(
            model, trainset[sub], loss_fn=loss_fn, optimizer=optimizer,
            batch_size=batch_size, device=DEVICE)
        print(f"Epoch: {epoch}, Subject {subjects[idx_train[sub]]}, {sub+1}/{n_train}, "
              f"Train accu: {train_accu_sub:.3f}, loss: {train_loss_sub:.3f}")
        train_accu[epoch, sub] = train_accu_sub
        train_loss[epoch, sub] = train_loss_sub
       
    for sub in range(n_valid):
        valid_accu_sub, valid_loss_sub = evaluate(
            model, validset[sub], loss_fn=loss_fn, 
            batch_size=batch_size, device=DEVICE)
        print(f"Epoch: {epoch}, Subject {subjects[idx_valid[sub]]}, {sub+1}/{n_valid}, "
              f"Valid accu: {valid_accu_sub:.3f}, loss: {valid_loss_sub:.3f}")
        valid_accu[epoch, sub] = valid_accu_sub
        valid_loss[epoch, sub] = valid_loss_sub

    for sub in range(n_test):
        test_accu_sub, test_loss_sub = evaluate(
            model, testset[sub], loss_fn=loss_fn, 
            batch_size=batch_size, device=DEVICE)
        print(f"Epoch: {epoch}, Subject {subjects[idx_test[sub]]}, {sub+1}/{n_test}, "
              f"Test accu: {test_accu_sub:.3f}, loss: {test_loss_sub:.3f}")
        test_accu[epoch, sub] = test_accu_sub
        test_loss[epoch, sub] = test_loss_sub

    train_accu_epoch = np.mean(train_accu[epoch])
    train_loss_epoch = np.mean(train_loss[epoch])
    valid_accu_epoch = np.mean(valid_accu[epoch])
    valid_loss_epoch = np.mean(valid_loss[epoch])
    test_accu_epoch = np.mean(test_accu[epoch])
    test_loss_epoch = np.mean(test_loss[epoch])
    print(f"Epoch: {epoch}, "
          f"Train accu: {train_accu_epoch:.3f}, loss: {train_loss_epoch:.3f}, "
          f"Valid accu: {valid_accu_epoch:.3f}, loss: {valid_loss_epoch:.3f}, "
          f"Test accu: {test_accu_epoch:.3f}, loss: {test_loss_epoch:.3f}, "
          f"Epoch time = {time_since(start, 1/n_epochs, epoch/n_epochs)}")


import matplotlib.pyplot as plt
fig = plt.figure()
axL = fig.add_subplot(111)
plt.plot(np.mean(train_accu, axis=1)*100, '-r')
plt.plot(np.mean(valid_accu, axis=1)*100, '-g')
plt.plot(np.mean(test_accu, axis=1)*100, '-b')
plt.ylabel('Accuracy [%]')
plt.legend(['train accu', 'valid accu', 'test accu'])
axR = fig.add_subplot(111, sharex=axL, frameon=False)
axR.yaxis.tick_right()
axR.yaxis.set_label_position("right")
plt.plot(np.mean(train_loss, axis=1), '--r')
plt.plot(np.mean(valid_loss, axis=1), '--g')
plt.plot(np.mean(test_loss, axis=1), '--b')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.grid(which='both', axis='both')
plt.legend(['train loss', 'valid loss', 'test loss'])
plt.tight_layout()
plt.show()
