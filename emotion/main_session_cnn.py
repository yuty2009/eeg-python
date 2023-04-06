# -*- coding:utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
import sys; sys.path.append(os.path.dirname(__file__)+"/../")
from common.transforms import *
from common.torchutils import RememberBest
from common.stopcriteria import Or, MaxEpochs, NoIncrease, ColumnBelow
from common.torchutils import train_epoch, evaluate

import deapreader
import seedreader
from convnet import CSPNet, EEGNet, ShallowConvNet, DeepConvNet
from acrnn import ACRNN
from eegtransformer import EEGTransformer


setname = 'seed'
fs = 200
n_classes = 3
chanset = np.arange(62)
n_channels = len(chanset)
datapath = 'E:/eegdata/emotion/seed/Preprocessed_EEG/'
sessions = seedreader.get_session_list(datapath)

window = 5 * fs
timewin = [0, 5]
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

tf_eeg = Compose((ToTensor(), RandomTemporalShift()))

torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

outpath = os.path.join(datapath, 'output')
if not os.path.exists(outpath):
    os.makedirs(outpath)

n_epochs = 1500
n_epochs_full = 600
n_epochs_nochange = 200
batch_size = 64
monitor_items = [
    'train_accu', 'train_loss',
    'valid_accu', 'valid_loss',
    ]
train_after_earlystop = True

test_accus = np.zeros((len(sessions)))
for ss in range(len(sessions)):
    session = sessions[ss]
    print('Load EEG epochs for ' + session)
    data_sess, labels_sess = seedreader.get_session_data(datapath, session, chanset, window, window)
    # the first 9 trials for training
    data_train = data_sess[:9]
    labels_train = labels_sess[:9]
    data_train = np.concatenate(data_train)
    labels_train = np.concatenate(labels_train)
    trainfullset = seedreader.EEGDataset(data_train, labels_train, tf_eeg)
    # split the whole trainset into train and valid sets
    valid_fraction = 0.2
    trainfullset_size = len(trainfullset)
    validset_size = int(valid_fraction * trainfullset_size)
    trainset_size = trainfullset_size - validset_size
    trainset, validset = random_split(trainfullset, [trainset_size, validset_size])
    # the last 6 trials for testing
    data_test = data_sess[9:]
    labels_test = labels_sess[9:]
    data_test = np.concatenate(data_test)
    labels_test = np.concatenate(labels_test)
    testset = seedreader.EEGDataset(data_test, labels_test, tf_eeg)

    model = EEGTransformer(n_timepoints, n_channels, n_classes).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs_df = pd.DataFrame(columns=monitor_items)
    remember_best = RememberBest('valid_accu', order=-1)
    stop_criterion = Or( # for early stop
        [
            MaxEpochs(n_epochs),
            NoIncrease("valid_accu", n_epochs_nochange),
        ]
    )

    epoch = 0
    stop = False
    early_stop = False
    while not stop:
    # for epoch in range(1, n_epochs+1):
        epoch = epoch + 1
        start_time = time.time()

        train_accu, train_loss = train_epoch(
            model, trainset, criterion=criterion, optimizer=optimizer,
            batch_size=batch_size, device=device)
        valid_accu, valid_loss  = evaluate(model, validset, criterion=criterion, batch_size=batch_size, device=device)
        
        epochs_df = epochs_df.append({
            'train_accu': train_accu, 'train_loss': train_loss,
            'valid_accu': valid_accu, 'valid_loss': valid_loss,
        }, ignore_index=True)
        remember_best.remember_epoch(epochs_df, model, optimizer)
        stop = stop_criterion.should_stop(epochs_df)
        if stop:
            # first load the best model
            remember_best.reset_to_best_model(epochs_df, model, optimizer)
            # Now check if  we should continue training:
            if train_after_earlystop:
                if not early_stop:
                    stop = False
                    early_stop = True
                    print('Early stop reached now continuing with full trainset')
                    epoch = 0
                    epochs_df.drop(epochs_df.index, inplace=True)
                    trainset = trainfullset # Use the full train dataset
                    remember_best = RememberBest('valid_loss', order=1)
                    stop_criterion = Or( # for full trainset training
                        [
                            MaxEpochs(n_epochs_full),
                            ColumnBelow("valid_loss", train_loss),
                        ]
                    )

        test_accu, test_loss  = evaluate(model, testset, criterion=criterion, batch_size=batch_size, device=device)
        
        print((f"Epoch: {epoch}, "
            f"Train accu: {train_accu:.3f}, loss: {train_loss:.3f}, "
            f"Valid accu: {valid_accu:.3f}, loss: {valid_loss:.3f}, "
            f"Test accu:  {test_accu:.3f},  loss: {test_loss:.3f}, "
            f"Epoch time = {time.time() - start_time: .3f} s"))

    test_accus[ss] = test_accu

sessions.append('average')
test_accus = np.append(test_accus, np.mean(test_accus))
df_results = pd.DataFrame({'session': sessions, 'accuracy': test_accus})
df_results.to_csv(
    os.path.join(outpath, 'results_' + model._get_name() 
    + '_' + time.strftime('%Y%m%d%H%M%S.csv'))
)
print(f'Overall accuracy: {np.mean(test_accus): .3f}')

import matplotlib.pyplot as plt
x = np.arange(len(test_accus))
plt.bar(x, test_accus)
plt.title('Averaged accuracy for all subjects')
plt.xticks(x, sessions)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
