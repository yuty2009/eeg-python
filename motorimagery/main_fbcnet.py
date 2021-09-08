# -*- coding:utf-8 -*-

import time
import numpy as np
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from common.transforms import *
from common.torchutils import RememberBest
from common.stopcriteria import Or, MaxEpochs, NoIncrease, ColumnBelow
from common.torchutils import train_epoch, evaluate
from mireader import *
from fbcnet import FBCNet


"""
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
"""
setname = 'bcicomp2005IIIa'
fs = 250
n_classes = 4
chanset = np.arange(60)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2005/IIIa/'
subjects = ['k3', 'k6', 'l1']
"""

setname = 'bcicomp2008IIa'
fs = 250
n_classes = 4
chanset = np.arange(22)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2008/IIa/'
# datapath = '/Users/yuty2009/data/bcicompetition/bci2008/IIa/'
subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

order, fstart, fstop, fstep, ftrans = 4, 4, 40, 4, 2
f1s = np.arange(fstart, fstop, fstep)
f2s = np.arange(fstart+fstep, fstop+fstep, fstep)
fbanks = np.hstack((f1s[:,None], f2s[:,None]))
# fbanks = [[7, 30], [30, 40]]
n_fbanks = len(fbanks)

timewin = [0.5, 4.5] # 0.5s pre-task data
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

tf_tensor = ToTensor()

torch.manual_seed(20190821)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

outpath = os.path.join(datapath, 'output')
if not os.path.exists(outpath):
    os.makedirs(outpath)

n_epochs = 1500
n_epochs_full = 600
n_epochs_nochange = 200
batch_size = 16
monitor_items = [
    'train_accu', 'train_loss',
    'valid_accu', 'valid_loss',
    ]
train_after_earlystop = True

test_accus = np.zeros(len(subjects))
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    print('Extract multi-band features from epochs for subject ' + subject)
    featTrain_bands = []
    featTest_bands = []
    for k in range(n_fbanks):
        f1, f2 = fbanks[k]
        fpass = [f1*2.0/fs, f2*2.0/fs]
        fstop =  [(f1-ftrans)*2.0/fs, (f2+ftrans)*2.0/fs]
        # fb, fa = signal.butter(order, fpass, btype='bandpass')
        fb, fa = signal.cheby2(order, 30, fstop, btype='bandpass')
        featTrain, labelTrain = extract_rawfeature(dataTrain, targetTrain, sampleseg, chanset, [fb, fa])
        featTest, labelTest = extract_rawfeature(dataTest, targetTest, sampleseg, chanset, [fb, fa])
        featTrain_bands.append(featTrain)
        featTest_bands.append(featTest)
    featTrain_bands = np.transpose(np.stack(featTrain_bands, axis=0), [1, 2, 3, 0]) # (n, t, c, b)
    featTest_bands = np.transpose(np.stack(featTest_bands, axis=0), [1, 2, 3, 0])
    n_train = featTrain.shape[1]
    n_test = featTest.shape[1]

    trainset_full = EEGDataset(featTrain_bands, labelTrain, tf_tensor)
    testset = EEGDataset(featTest_bands, labelTest, tf_tensor)

    valid_set_fraction = 0.2
    valid_set_size = int(len(trainset_full) * valid_set_fraction)
    train_set_size = len(trainset_full) - valid_set_size
    trainset, validset = random_split(trainset_full, [train_set_size, valid_set_size])
    # trainset = Subset(trainset_full, list(range(0, train_set_size)))
    # validset = Subset(trainset_full, list(range(train_set_size, len(trainset_full))))

    model = FBCNet(n_timepoints, n_channels, n_classes, n_fbanks).to(device)
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
        valid_accu, valid_loss = evaluate(model, validset, criterion=criterion, batch_size=batch_size, device=device)
        
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
                    trainset = trainset_full # Use the full train dataset
                    remember_best = RememberBest('valid_loss', order=1)
                    stop_criterion = Or( # for full trainset training
                        [
                            MaxEpochs(n_epochs_full),
                            ColumnBelow("valid_loss", train_loss),
                        ]
                    )

        test_accu, test_loss = evaluate(model, testset, criterion=criterion, batch_size=batch_size, device=device)
        
        print((f"Epoch: {epoch}, "
               f"Train accu: {train_accu:.3f}, loss: {train_loss:.3f}, "
               f"Valid accu: {valid_accu:.3f}, loss: {valid_loss:.3f}, "
               f"Test accu:  {test_accu:.3f},  loss: {test_loss:.3f}, "
               f"Epoch time = {time.time() - start_time: .3f} s"))

        test_accus[ss] = test_accu

print(f'Overall accuracy: {np.mean(test_accus): .3f}')

import matplotlib.pyplot as plt
x = np.arange(len(test_accus))
plt.bar(x, test_accus)
plt.title('Averaged accuracy for all subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()
