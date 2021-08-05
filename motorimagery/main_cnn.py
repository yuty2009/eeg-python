# -*- coding:utf-8 -*-

import numpy as np
from torch.utils.data import random_split
from common.linear import *
from common.torchutils import RememberBest
from common.stopcriteria import Or, MaxEpochs, NoIncrease
from common.torchutils import train_epoch, evaluate, DEVICE
from motorimagery.mireader import *
from motorimagery.convnet import CSPNet, EEGNet, ShallowConvNet, DeepConvNet



setname = 'bcicomp2005IVa'
fs = 100
n_classes = 2
chanset = np.arange(118)
# chanset = [
#     np.arange(13,22),
#     np.arange(32,39),
#     np.arange(49,58),
#     np.arange(67,76),
#     np.arange(86,95),
#     np.array([103, 105, 107, 111, 112, 113])
#     ]
# chanset = np.hstack(chanset)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2005/IVa/'
subjects = ['aa', 'al', 'av', 'aw', 'ay']

"""
setname = 'bcicomp2005IIIa'
fs = 250
n_classes = 4
chanset = np.arange(60)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2005/IIIa/'
subjects = ['k3', 'k6', 'l1']
"""
"""
setname = 'bcicomp2008IIa'
fs = 250
n_classes = 4
chanset = np.arange(22)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2008/IIa/'
# datapath = '/Users/yuty2009/data/bcicompetition/bci2008/IIa/'
subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
"""

f1 = 4
f2 = 38
order = 3
fb, fa = signal.butter(order, [2*f1/fs, 2*f2/fs], btype='bandpass')
# fb, fa = signal.cheby2(order, 40, [2*f1/fs, 2*f2/fs], btype='bandpass')
# show_filter(fb, fa, fs)

timewin = [0.5, 4.5] # 0.5s pre-task data
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

n_epochs = 500
n_epochs_increase = 150
batch_size = 64
monitor_items = [
    'train_accu', 'train_loss',
    'valid_accu', 'valid_loss',
    ]

torch.manual_seed(4)
tf_epoch = TransformEpoch()

accTest = np.zeros(len(subjects))
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    print('Extract raw features from epochs for subject ' + subject)
    fTrain, labelTrain = extract_rawfeature(dataTrain, targetTrain, [fb, fa], sampleseg, chanset)
    fTest, labelTest = extract_rawfeature(dataTest, targetTest, [fb, fa], sampleseg, chanset)
    trainset = EEGDataset(fTrain, labelTrain, tf_epoch)
    testset = EEGDataset(fTest, labelTest, tf_epoch)

    valid_set_fraction = 0.1
    valid_set_size = int(len(trainset) * valid_set_fraction)
    train_set_size = len(trainset) - valid_set_size
    trainset, validset = random_split(trainset, [train_set_size, valid_set_size])

    model = CSPNet(n_timepoints, n_channels, n_classes).to(DEVICE)
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)

    epochs_df = pd.DataFrame(columns=monitor_items)
    remember_best = RememberBest('valid_accu')
    stop_criterion = Or(
        [
            MaxEpochs(n_epochs),
            NoIncrease("valid_accu", n_epochs_increase),
        ]
    )

    for epoch in range(1, n_epochs+1):
        start = time.time()

        train_accu, train_loss = train_epoch(
            model, trainset, loss_fn=loss_fn, optimizer=optimizer,
            batch_size=batch_size, device=DEVICE)
        valid_accu, valid_loss  = evaluate(model, validset, loss_fn=loss_fn, batch_size=batch_size, device=DEVICE)
        
        epochs_df = epochs_df.append({
            'train_accu': train_accu, 'train_loss': train_loss,
            'valid_accu': valid_accu, 'valid_loss': valid_loss,
        }, ignore_index=True)
        remember_best.remember_epoch(epochs_df, model, optimizer)

        test_accu, test_loss  = evaluate(model, testset, loss_fn=loss_fn, batch_size=batch_size, device=DEVICE)
        
        print((f"Epoch: {epoch}, "
               f"Train accu: {train_accu:.3f}, loss: {train_loss:.3f}, "
               f"Valid accu: {valid_accu:.3f}, loss: {valid_loss:.3f}, "
               f"Test accu:  {test_accu:.3f},  loss: {test_loss:.3f}, "
               f"Epoch time = {time_since(start, 1/n_epochs, epoch/n_epochs)}"))

        if stop_criterion.should_stop(epochs_df): break

    accTest[ss] = test_accu

print(np.mean(accTest))

import matplotlib.pyplot as plt
x = np.arange(len(accTest))
plt.bar(x, accTest*100)
plt.title('Accuracy for the five subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
