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


def get_subject_list(datadir):
    session_list = get_session_list(datadir)
    subject_list = [session_name.split('_')[0] for session_name in session_list]
    subject_list = np.unique(subject_list)
    return subject_list


def get_session_list(datadir):
    session_list = []
    for filepath in glob.glob(datadir + "/*.mat"):
        if os.path.basename(filepath) == 'label.mat': continue
        session_list.append(os.path.basename(filepath)[:-4])
    return session_list


def get_subject_data(datadir, subject, chanset, window, stride):
    data_sub = []
    labels_sub = []
    labelpath = datadir + '/label.mat'
    for filepath in glob.glob(datadir + "/" + subject + "*.mat"):
        data_1, labels_1 = load_session_data(filepath, labelpath, chanset, window, stride)
        # data_1 = np.concatenate(data_1, axis=0)
        # labels_1 = np.concatenate(labels_1, axis=0)
        data_sub.append(data_1)
        labels_sub.append(labels_1)
    # data_sub = np.concatenate(data_sub, axis=0)
    # labels_sub = np.concatenate(labels_sub, axis=0)
    return data_sub, labels_sub


def get_session_data(datadir, session, chanset, window, stride):
    labelpath = datadir + '/label.mat'
    filepath = datadir + '/' + session + '.mat'
    return load_session_data(filepath, labelpath, chanset, window, stride)


def load_session_data(datapath, labelpath, chanset, window, stride):
    data_dict = sio.loadmat(datapath)
    data_keys = list(data_dict.keys())[3:]
    data_list = [data_dict[key] for key in data_keys]
    labels = sio.loadmat(labelpath)['label'][0] + 1 # change labels from [-1, 0, 1] to [0, 1, 2]
    return extract_eegdata(data_list, labels, chanset, window, stride)


def extract_eegdata(data, labels, chanset=None, window=1000, stride=None):
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
        data_trial = np.transpose(data_trial, [0, 2, 1]) # n_samples, n_timepoints, n_channels
        labels_trial = np.array(labels_trial)
        data_extracted.append(data_trial)
        labels_extracted.append(labels_trial)
    return data_extracted, labels_extracted


def get_session_feature(datadir, session, featname):
    labelpath = datadir + '/label.mat'
    filepath = datadir + '/' + session + '.mat'
    return load_session_feature(filepath, labelpath, featname)


def load_session_feature(datapath, labelpath, featname):
    labels = sio.loadmat(labelpath)['label'][0] + 1 # change labels from [-1, 0, 1] to [0, 1, 2]
    data_dict = sio.loadmat(datapath)
    data_keys = list(data_dict.keys())[3:]
    feature_list = []
    labels_list = []
    for key in data_keys:
        if featname in key:
            data_trial = data_dict[key]
            data_trial = np.transpose(data_trial, [1, 2, 0])
            n_samples = data_trial.shape[0]
            label_index = int(key[len(featname):]) - 1
            label_trial = labels[label_index] * np.ones([n_samples], dtype=np.int32)
            feature_list.append(data_trial)
            labels_list.append(label_trial)
    return feature_list, labels_list


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


if __name__ == '__main__':

    datapath = 'e:/eegdata/emotion/seed/Preprocessed_EEG/'
    featurepath = 'e:/eegdata/emotion/seed/ExtractedFeatures/'
    labelpath = datapath + 'label.mat'
    featname = 'de_LDS'

    fs = 200
    eegchan = np.arange(62)
    window = int(5.0 * fs)
    stride = window
    fbands = [4, 8, 13, 30, 47] # theta, alpha, beta, gamma

    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    subject_list = get_subject_list(datapath)
    session_list = get_session_list(datapath)

    feature_sess, labels_sess = get_session_feature(featurepath, session_list[0], featname)

    data_all = []
    for filepath in glob.glob(datapath + "/*.mat"):
        if filepath == labelpath: continue
        print('Load and extract feature for %s' % filepath)
        data, target = load_session_data(filepath, labelpath, eegchan, window, stride)
        features, labels = extract_feature_bandpower(data, target, fbands)
        data_i = dict()
        data_i['features'] = features
        data_i['labels'] = labels
        data_all.append(data_i)
    np.savez(datapath + 'processed/data_bandpower.npz',data=data_all)

