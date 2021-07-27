# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from common.datawrapper import read_matdata
from common.signalproc import *
from common.spatialfilter import *


# translate epoch data (n, c, t) into grayscale images (n, 1, c, t)
class TransformEpoch(object):
    def __call__(self, epoch):
        epoch = torch.Tensor(epoch)
        return torch.unsqueeze(epoch, dim=0)


class EEGDataset(Dataset):
    def __init__(self, epochs, labels, transforms=None):
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.labels = torch.Tensor(labels - 1).long() # label {1, 2} to {0, 1}

    def __getitem__(self, idx):
        return self.epochs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_eegdata(filepath, labelpath):
    data = read_matdata(filepath, ['cnt', 'mrk', 'nfo'])
    labeldata = read_matdata(labelpath, ['true_y'])
    cnt = 0.1*data['cnt']
    pos = data['mrk']['pos'][0,0][0]
    code = data['mrk']['y'][0,0][0]
    fs = data['nfo']['fs'][0,0][0]
    label = labeldata['true_y'][0]

    code[np.argwhere(np.isnan(code))] = 0
    num_train = len(np.argwhere(code >= 1))
    num_test = 280 - num_train
    timewin = [-0.5, 4.0]
    sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
    num_samples = sampleseg[1] - sampleseg[0]
    num_channels = cnt.shape[1]

    targetTrain = np.zeros(num_train)
    dataTrain = np.zeros([num_train, num_samples, num_channels])
    for i in range(num_train):
        begin = pos[i] + sampleseg[0]
        end = begin + num_samples
        dataTrain[i,:,:] = cnt[begin:end,:]
        targetTrain[i] = code[i]

    targetTest = np.zeros(num_test)
    dataTest = np.zeros([num_test, num_samples, num_channels])
    for i in range(num_test):
        begin = pos[num_train+i]
        end = begin + num_samples
        dataTest[i,:,:] = cnt[begin:end,:]
        targetTest[i] = label[num_train+i]

    return dataTrain, targetTrain, dataTest, targetTest


def extract_rawfeature(data, target, filter, sampleseg, chanset):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_samples_used = sample_end - sample_begin
    num_channel_used = len(chanset)

    # show_filtering_result(filter[0], filter[1], data[0,:,0])

    labels = target
    features = np.zeros([num_trials, num_samples_used, num_channel_used])
    for i in range(num_trials):
        signal_epoch = data[i]
        signal_filtered = signal_epoch
        for j in range(num_channels):
            signal_filtered[:, j] = signal.filtfilt(filter[0], filter[1], signal_filtered[:, j])
        features[i] = signal_filtered[sample_begin:sample_end, chanset]

    return features, labels


def extract_variance(data, target, filter, sampleseg, chanset):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_samples_used = sample_end - sample_begin
    num_channel_used = len(chanset)

    # show_filtering_result(filter[0], filter[1], data[0,:,0])

    labels = target
    Rs = np.zeros([num_trials, num_channel_used, num_channel_used])
    for i in range(num_trials):
        signal_epoch = data[i]
        signal_filtered = signal_epoch
        for j in range(num_channels):
            signal_filtered[:, j] = signal.filtfilt(filter[0], filter[1], signal_filtered[:, j])
        signal_filtered = signal_filtered[sample_begin:sample_end, chanset]
        Rs[i] = np.dot(signal_filtered.T, signal_filtered)/np.trace(np.dot(signal_filtered.T, signal_filtered))

    return Rs, labels


def extract_variance_multiband(data, target, bands, sampleseg, chanset):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_samples_used = sample_end - sample_begin
    num_channel_used = len(chanset)

    fs = 100
    order = 5
    num_bands = len(bands)
    Rss = []
    for k in range(num_bands):
        f1, f2 = bands[k]
        # fb, fa = signal.butter(order, [2*f1/fs, 2*f2/fs], btype='bandpass')
        fb, fa = signal.cheby2(order, 40, [2*f1/fs, 2*f2/fs], btype='bandpass')
        Rs = np.zeros([num_trials, num_channel_used, num_channel_used])
        for i in range(num_trials):
            signal_epoch = data[i]
            signal_filtered = signal_epoch
            for j in range(num_channels):
                signal_filtered[:, j] = signal.filtfilt(fb, fa, signal_filtered[:, j])
            signal_filtered = signal_filtered[sample_begin:sample_end, chanset]
            Rs[i] = np.dot(signal_filtered.T, signal_filtered)/np.trace(np.dot(signal_filtered.T, signal_filtered))
        Rss.append(Rs)

    labels = target

    return Rss, labels


def load_dataset(datapath, subject):
    f = np.load(datapath+'processed/'+subject+'.npz')
    return f['dataTrain'], f['targetTrain'], f['dataTest'], f['targetTest']


if __name__ == '__main__':

    datapath = 'E:/bcicompetition/bci2005/IVa/'
    subjects = ['aa', 'al', 'av', 'aw', 'ay']

    import os
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    for ss in range(len(subjects)):
        subject = subjects[ss]
        filepath = datapath+'data_set_IVa_'+subject+'.mat'
        labelpath = datapath+'true_labels_'+subject+'.mat'

        print('Load and extract continuous EEG into epochs for subject '+subject)
        dataTrain, targetTrain, dataTest, targetTest = load_eegdata(filepath, labelpath)

        np.savez(datapath+'processed/'+subject+'.npz',
                 dataTrain=dataTrain, targetTrain=targetTrain, dataTest=dataTest, targetTest=targetTest)


