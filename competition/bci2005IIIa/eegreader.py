# -*- coding: utf-8 -*-

import os
import numpy as np
from common.datawrapper import *
from common.spatialfilter import *
from common.temporalfilter import *


def load_eegdata(filepath, labelpath):
    label_key = os.path.basename(labelpath)
    label_key = label_key[:-4]
    labeldata = read_matdata(labelpath, [label_key])
    label = labeldata[label_key]

    data = read_matdata(filepath, ['s', 'HDR'])
    s = data['s']
    fs = data['HDR']['SampleRate'][0][0][0][0]
    pos = data['HDR']['TRIG'][0][0].squeeze()
    code = data['HDR']['Classlabel'][0][0].squeeze()
    index_train = np.squeeze(np.argwhere(~np.isnan(code)))
    index_test = np.squeeze(np.argwhere(np.isnan(code)))

    num_train = len(index_train)
    num_test = len(index_test)
    num_samples = 4 * fs # 4s data
    num_channels = s.shape[1]

    # There are NaNs in the data. We replace them by interpolation.
    for c in range(num_channels):
        sc = s[:, c]
        nans = np.isnan(sc)
        inds = np.arange(len(sc))
        if any(nans):
            sc[nans] = np.interp(inds[nans], inds[~nans], sc[~nans])
            s[:, c] = sc

    targetTrain = np.zeros(num_train, dtype=np.int)
    dataTrain = np.zeros([num_train, num_samples, num_channels])
    for i in range(num_train):
        ii = index_train[i]
        begin = pos[ii] + 3*fs # 3 seconds prepare
        end = begin + num_samples
        dataTrain[i,:,:] = s[begin:end,:]
        targetTrain[i] = code[ii]

    targetTest = np.zeros(num_test, dtype=np.int)
    dataTest = np.zeros([num_test, num_samples, num_channels])
    for i in range(num_test):
        ii = index_test[i]
        begin = pos[ii] + 3*fs # 3 seconds prepare
        end = begin + num_samples
        dataTest[i,:,:] = s[begin:end,:]
    targetTest = np.squeeze(label[index_test]).astype(np.int)

    return dataTrain, targetTrain, dataTest, targetTest


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

    fs = 250
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

    datapath = 'E:/bcicompetition/bci2005/IIIa/'
    subjects = ['k3', 'k6', 'l1']

    import os
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    for ss in range(len(subjects)):
        subject = subjects[ss]
        filepath = datapath+subject+'b.mat'
        labelpath = datapath+'true_label_'+subject+'.mat'

        print('Load and extract continuous EEG into epochs for subject '+subject)
        dataTrain, targetTrain, dataTest, targetTest = load_eegdata(filepath, labelpath)

        np.savez(datapath+'processed/'+subject+'.npz',
                 dataTrain=dataTrain, targetTrain=targetTrain, dataTest=dataTest, targetTest=targetTest)


