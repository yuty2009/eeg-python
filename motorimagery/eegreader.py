# -*- coding: utf-8 -*-

import numpy as np
from common.datawrapper import *
from common.spatialfilter import *
from common.temporalfilter import *


def load_eegdata(filepath, labelpath):
    data = read_matdata(filepath, ['cnt', 'mrk'])
    labeldata = read_matdata(labelpath, ['true_y'])
    cnt = 0.1*data['cnt']
    pos = data['mrk']['pos'][0,0][0]
    code = data['mrk']['y'][0,0][0]
    label = labeldata['true_y'][0]

    code[np.argwhere(np.isnan(code))] = 0
    num_train = len(np.argwhere(code >= 1))
    num_test = 280 - num_train
    num_samples = 350
    num_channels = cnt.shape[1]

    targetTrain = np.zeros(num_train)
    dataTrain = np.zeros([num_train, num_samples, num_channels])
    for i in range(num_train):
        begin = pos[i] - 1
        end = begin + num_samples
        dataTrain[i,:,:] = cnt[begin:end,:]
        targetTrain[i] = code[i]

    targetTest = np.zeros(num_test)
    dataTest = np.zeros([num_test, num_samples, num_channels])
    for i in range(num_test):
        begin = pos[num_train+i] - 1
        end = begin + num_samples
        dataTest[i,:,:] = cnt[begin:end,:]
        targetTest[i] = label[num_train+i]

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
            signal_filtered[:, j] = filtfilt(filter[0], filter[1], signal_filtered[:, j])
        signal_filtered = signal_filtered[sample_begin:sample_end, chanset]
        Rs[i] = np.dot(signal_filtered.T, signal_filtered)/np.trace(np.dot(signal_filtered.T, signal_filtered))

    return Rs, labels


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


