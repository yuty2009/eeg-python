# -*- coding: utf-8 -*-

import pickle
import numpy as np
from common.datawrapper import *
from common.signalproc import *
from common.timefreq import *
from scipy.stats import zscore


def load_eegdata(filepath, chanset, stride, window):
    f = open(filepath, 'rb')  # Read the file in Binary mode
    x = pickle.load(f, encoding='latin1')
    data, target = x['data'], x['labels']
    return extract_eegdata(data, target, chanset, stride, window)


def extract_eegdata(data, target, chanset, stride, window):
    fs = 128
    num_samples = 60 * fs
    target[target<4.5] = 0
    target[target>=4.5] = 1
    data_used = data[:, chanset, 3*fs:]

    start = 0
    data_extracted = []
    labels_extracted = []
    while start + window <= num_samples:
        data_seg = data_used[:, :, start:start+window]
        data_extracted.append(data_seg)
        labels_extracted.append(target)
        start += stride
    data_extracted = np.concatenate(data_extracted, axis=0)
    labels_extracted = np.concatenate(labels_extracted, axis=0)
    return data_extracted, labels_extracted


def extract_feature_bandpower(data, labels, fbands):
    fs = 128
    num_examples, num_channels, num_samples = data.shape
    num_features = len(fbands) - 1
    features = np.zeros((num_examples, num_channels, num_features))
    for i in range(num_examples):
        for j in range(num_channels):
            signal_ij = data[i, j, :]
            features[i, j, :] = bandpower(signal_ij, fs, fbands)
    features = np.reshape(features, [num_examples, -1])
    return features, labels


def load_dataset(datapath):
    f = np.load(datapath+'processed/data_bandpower.npz', allow_pickle=True)
    return f['data']


if __name__ == '__main__':

    datapath = 'e:/eegdata/emotion/deap/data_preprocessed_python/'
    # datapath = '/home/yuty2009/data/eegdata/emotion/deap/data_preprocessed_python/'

    fs = 128
    eegchan = np.arange(32)
    stride = int(1.0 * fs)
    window = int(2.0 * fs)
    fbands = [4, 8, 13, 30, 47] # theta, alpha, beta, gamma

    import os
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    import glob
    data_all = []
    for filepath in glob.glob(datapath + "/*.dat"):
        print('Load and extract feature for %s' % filepath)
        data, target = load_eegdata(filepath, eegchan, stride, window)
        features, labels = extract_feature_bandpower(data, target, fbands)
        data_i = dict()
        data_i['features'] = features
        data_i['labels'] = labels
        data_all.append(data_i)
    np.savez(datapath + 'processed/data_bandpower.npz',data=data_all)

