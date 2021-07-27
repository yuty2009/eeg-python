# -*- coding: utf-8 -*-

import os
import numpy as np
from common.datawrapper import *
from common.signalproc import *
from common.spatialfilter import *


def load_eegdata(fdatatrain, flabeltrain, fdatatest, flabeltest):
    s, events, clabs = read_gdfdata(fdatatrain)
    pos = events['pos']
    code = events['type']
    indices = np.argwhere(code == 768)

    num_train = len(indices)
    fs = 250
    num_samples = 4 * fs # 4s data
    num_channels = s.shape[1]
    dataTrain = np.zeros([num_train, num_samples, num_channels])
    for i in range(num_train):
        begin = pos[indices[i, 0]] + 2*fs # 2 seconds prepare
        end = begin + num_samples
        dataTrain[i,:,:] = s[begin:end,:]
    labeldata = read_matdata(flabeltrain, ['classlabel'])
    targetTrain = np.array(np.squeeze(labeldata['classlabel']), dtype=np.int)

    s, events, clabs = read_gdfdata(fdatatest)
    pos = events['pos']
    code = events['type']
    indices = np.argwhere(code == 768)

    num_test = len(indices)
    dataTest = np.zeros([num_test, num_samples, num_channels])
    for i in range(num_test):
        begin = pos[indices[i, 0]] + 2*fs # 2 seconds prepare
        end = begin + num_samples
        dataTest[i,:,:] = s[begin:end,:]
    labeldata = read_matdata(flabeltest, ['classlabel'])
    targetTest = np.array(np.squeeze(labeldata['classlabel']), dtype=np.int)
    
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

    datapath = 'E:/bcicompetition/bci2008/IIa/'
    subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

    import os
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    for ss in range(len(subjects)):
        subject = subjects[ss]
        fdatatrain = datapath+subject+'T.gdf'
        flabeltrain = datapath+subject+'T.mat'
        fdatatest = datapath+subject+'E.gdf'
        flabeltest = datapath+subject+'E.mat'

        print('Load and extract continuous EEG into epochs for subject '+subject)
        dataTrain, targetTrain, dataTest, targetTest = load_eegdata(fdatatrain, flabeltrain, fdatatest, flabeltest)

        np.savez(datapath+'processed/'+subject+'.npz',
                 dataTrain=dataTrain, targetTrain=targetTrain, dataTest=dataTest, targetTest=targetTest)


