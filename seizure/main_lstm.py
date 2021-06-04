# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.utils import *
from common.datawrapper import *
from seizure.eegreader_csv import *


class RNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(RNNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=15, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
            # nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
            # nn.BatchNorm1d(64)
        )
        self.lstm1 = nn.LSTM(64, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute((0, 2, 1))
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :] # the last output of lstm
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


datapath = 'e:/eegdata/seizure/Epileptic_Seizure_Recognition/data.csv'
data, labels = read_csvdata(datapath)
data = np.expand_dims(data, axis=1)
dataset = Dataset(data, labels)
trainset, validset = dataset.get_subset([80, 20])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNNModel(1, num_classes=5).to(device)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

epochs = 400
batch_size = 128
num_batches = np.ceil(trainset.num_examples / batch_size).astype('int')
for epoch in range(epochs):
    acc_train = 0
    loss_train = 0
    start = time.time()
    for step in range(num_batches):
        X_batch, y_batch = trainset.next_batch(batch_size)
        X_batch = torch.tensor(X_batch, device=device, dtype=torch.float32)
        y_batch = torch.tensor(y_batch, device=device, dtype=torch.int64)

        yp_batch = model(X_batch)
        loss = F.cross_entropy(yp_batch, y_batch, reduction='sum')
        loss_train += loss.item()
        acc_train += (yp_batch.argmax(1) == y_batch).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    acc_train = acc_train / trainset.num_examples

    with torch.no_grad():
        features_valid, labels_valid = validset.features, validset.labels
        X_valid = torch.tensor(features_valid, device=device, dtype=torch.float32)
        y_valid = torch.tensor(labels_valid, device=device, dtype=torch.int64)
        yp_test = model(X_valid)
        loss_valid = F.cross_entropy(yp_test, y_valid, reduction='sum')
        acc_valid = (yp_test.argmax(1) == y_valid).sum().item() / validset.num_examples

    # print statistics every 100 steps
    if epoch > 0 and epoch % 1 == 0:
        print('Epoch: %d, ' % epoch, "lr : %f" % optimizer.param_groups[0]['lr'],
        "| %s" % time_since(start, epoch/ epochs))
        print(f'\tLoss: {loss_train:.4f}(train)\t|\tAcc: {acc_train * 100:.1f}%(train)')
        print(f'\tLoss: {loss_valid:.4f}(valid)\t|\tAcc: {acc_valid * 100:.1f}%(valid)')

