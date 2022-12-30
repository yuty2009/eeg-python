# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from common.timefreq import *


class EEGDataset(Dataset):
    def __init__(self, epochs, labels, transforms=None):
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.labels = torch.Tensor(labels).long()

    def __getitem__(self, idx):
        return self.epochs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_eegdata(datapath, labelpath, chanset=None, window=1000, stride=None):
    data_dict = sio.loadmat(datapath)
    data_keys = list(data_dict.keys())[3:]
    data_list = [data_dict[key] for key in data_keys]
    labels = sio.loadmat(labelpath)['label'][0] + 1
    return extract_eegdata(data_list, labels, chanset, window, stride)


def extract_eegdata(data, labels, chanset, window, stride):
    fs = 200
    if stride is None:
        stride = window
    if chanset is None:
        chanset = range(0, data[0].shape[0])

    data_extracted = []
    labels_extracted = []
    for i in range(len(data)):
        start = 0
        data_trial = []
        labels_trial = []
        data_used = data[i][chanset, ]
        while start + window < data_used.shape[1]:
            data_seg = data_used[:, start:start+window]
            data_trial.append(data_seg)
            labels_trial.append(labels[i])
            start += stride
        data_trial = np.array(data_trial)
        labels_trial = np.array(labels_trial)
        data_extracted.append(data_trial)
        labels_extracted.append(labels_trial)
    return data_extracted, labels_extracted


def extract_feature_bandpower(data, labels, fbands):
    fs = 200
    num_trials = len(data)
    num_features = len(fbands) - 1
    features = []
    for trial in range(num_trials):
        num_examples, num_channels, num_samples = data[trial].shape
        features_trial = np.zeros((num_examples, num_channels, num_features))
        for i in range(num_examples):
            for j in range(num_channels):
                signal_ij = data[trial][i, j, :]
                features_trial[i, j, :] = bandpower(signal_ij, fs, fbands)
        features_trial = np.reshape(features_trial, [num_examples, -1])
        features.append(features_trial)
    return features, labels


def load_dataset(datapath):
    f = np.load(datapath+'processed/data_bandpower.npz', allow_pickle=True)
    return f['data']


def get_subject_data(datadir, subject, chanset, window, stride):
    data_sub = []
    labels_sub = []
    labelpath = datadir + '/label.mat'
    for filepath in glob.glob(datadir + "/" + subject + "*.mat"):
        data_1, labels_1 = load_eegdata(filepath, labelpath, chanset, window, stride)
        data_1 = np.concatenate(data_1, axis=0)
        labels_1 = np.concatenate(labels_1, axis=0)
        data_sub.append(data_1)
        labels_sub.append(labels_1)
    data_sub = np.concatenate(data_sub, axis=0)
    data_sub = np.transpose(data_sub, [0, 2, 1]) # n_samples, n_timepoints, n_channels
    labels_sub = np.concatenate(labels_sub, axis=0)
    return data_sub, labels_sub


if __name__ == '__main__':

    datapath = 'e:/eegdata/emotion/seed/Preprocessed_EEG/'
    labelpath = datapath + 'label.mat'

    fs = 200
    eegchan = np.arange(62)
    window = int(5.0 * fs)
    stride = window
    fbands = [4, 8, 13, 30, 47] # theta, alpha, beta, gamma

    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    data_all = []
    for filepath in glob.glob(datapath + "/*.mat"):
        if filepath == labelpath: continue
        print('Load and extract feature for %s' % filepath)
        data, target = load_eegdata(filepath, labelpath, eegchan, window, stride)
        features, labels = extract_feature_bandpower(data, target, fbands)
        data_i = dict()
        data_i['features'] = features
        data_i['labels'] = labels
        data_all.append(data_i)
    np.savez(datapath + 'processed/data_bandpower.npz',data=data_all)

