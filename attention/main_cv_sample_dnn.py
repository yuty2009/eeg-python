# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import attention.eegreader_xml as ds
from attention.densenet import *
from attention.wavenet import *


datapath = 'e:/eegdata/attention/Headband/data4class/'
# datapath = '/home/yuty2009/data/eegdata/attention/Headband/data4class/'
modelpath = os.path.join(datapath, 'models/')
data = ds.load_dataset(datapath)
labels_all = []
features_all = []
for data_sub in data:
    labels_sub = data_sub['labels']
    # make it a binary classification problem
    # labels_sub[np.argwhere(labels_sub != 1)] = 0
    labels_all.append(labels_sub)
    features_sub = data_sub['features']
    # features_sub = np.reshape(features_sub, [features_sub.shape[0], -1])
    features_all.append(features_sub)
labels_all = np.concatenate(labels_all, axis=0)
features_all = np.concatenate(features_all, axis=0)
dataset = ds.Dataset(features_all, labels_all)

# feature_dim = features_all.shape[1]
in_planes, feature_dim = features_all.shape[-2:]

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
torch.cuda.manual_seed(42) if cuda else torch.manual_seed(42)


class Softmax(nn.Sequential):
    def __init__(self, num_features, num_classes=10):
        super(Softmax, self).__init__(
            nn.Linear(num_features, num_classes)
        )


# model = Softmax(feature_dim, 2).to(device)
# model = DenseNet_1d(in_planes=in_planes, feature_dim=feature_dim,
#                     init_planes=16, growth_rate=4,
#                     block_config=[4], num_classes=2).to(device)
model = MultiScaleNet(4, in_planes, 16, feature_dim, num_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.08)

num_folds = 10
dataset.set_num_folds(num_folds)
accs = np.zeros(num_folds)
for fold in range(num_folds):
    features_train, labels_train, features_test, labels_test = dataset.next_fold()
    trainset = ds.Dataset(features_train, labels_train)

    # train
    epochs = 100
    batch_size = 50
    verbose = True
    num_batches = np.ceil(trainset.num_examples / batch_size).astype('int')
    weight_reg = 1.0 / num_batches
    for epoch in range(epochs):
        for step in range(num_batches):
            X_batch, y_batch = trainset.next_batch(batch_size)
            X_batch = torch.tensor(X_batch, device=device, dtype=torch.float32)
            y_batch = torch.tensor(y_batch, device=device, dtype=torch.int64)

            yp_batch, loss_reg = model(X_batch)
            loss_ce = F.cross_entropy(yp_batch, y_batch, reduction='sum')
            loss = loss_ce + weight_reg * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics every 100 steps
            if verbose and (step + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss CE {:.4f}, Loss Reg {:.4f}"
                      .format(epoch + 1, epochs,
                              (step + 1) * batch_size, trainset.num_examples,
                              loss_ce.item(), loss_reg.item()))

    with torch.no_grad():
        X_test = torch.tensor(features_test, device=device, dtype=torch.float32)
        labels_predict, _ = model(X_test)
        labels_predict = labels_predict.argmax(dim=1)
        labels_predict = labels_predict.cpu().data.numpy()

    accs[fold] = np.mean(np.equal(labels_predict, labels_test).astype(np.float32))
    print("Fold %d, Accuracy %.4f" % (fold, accs[fold]))
print("Mean accuracy %.4f" % np.mean(accs))