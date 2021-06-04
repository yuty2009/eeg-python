# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import zscore
from common.datawrapper import *
from common.temporalfilter import *


def load_eegdata(filepath, filter):
    eegdata, events, clabs = read_cntdata(filepath)
    return extract_eegdata(eegdata, events, filter)


def extract_eegdata(eegdata, events, filter):
    fs = 250
    num_samples = int(0.6 * fs)
    num_channels = eegdata.shape[0]

    fb, fa = filter
    # show_filtering_result(fb, fa, signal[0,:,0])

    eventpos = events[:, 0]
    eventtype = events[:, 2]
    cuepos = np.logical_and(eventtype >= 41, eventtype <= 80).nonzero()[0]
    num_trials = len(cuepos) - 1 # The last character may not be complete.
    num_chars = len(np.unique(eventtype[cuepos[0]+1:cuepos[1]]))
    num_repeats = int((cuepos[1] - cuepos[0]) / num_chars)

    target = np.zeros(num_trials, dtype=int)
    data = np.zeros([num_trials, num_repeats, num_chars, num_samples, num_channels])
    for i in range(num_trials):
        target[i] = eventtype[cuepos[i]] - num_chars
        repeat = np.zeros(num_chars, dtype=int)
        for j in range(num_repeats):
            for k in range(num_chars):
                kk = cuepos[i] + 1 + j*num_chars + k
                event = eventtype[kk]
                if event > 0 and event <= num_chars:
                    repeat[event - 1] += 1
                    signal_epoch = eegdata[:, eventpos[kk]:eventpos[kk]+num_samples]
                    data[i, repeat[event-1]-1, event-1, :, :] = signal_epoch.T

    return data, target


def extract_feature(data, target, sampleseg, chanset, dfs):
    num_trials, num_repeats, num_chars, num_samples, num_channels = data.shape

    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_samples_used = int(np.ceil((sample_end - sample_begin) / dfs))
    num_channel_used = len(chanset)
    num_features = num_samples_used * num_channel_used

    np.seterr(divide='ignore', invalid='ignore')
    labels = np.zeros([num_trials, num_repeats, num_chars])
    feature = np.zeros([num_trials, num_repeats, num_chars, num_features])
    for trial in range(num_trials):
        labels[trial, :, target[trial]-1] = 1
        signal_trial = data[trial]
        for repeat in range(num_repeats):
            for char in range(num_chars):
                signal_epoch = signal_trial[repeat, char, :, :]
                signal_filtered = signal_epoch[sample_begin:sample_end, chanset]
                signal_downsampled = np.transpose(signal.decimate(signal_filtered.T, dfs, zero_phase=True))
                signal_normalized = np.zeros(signal_downsampled.shape)
                for c in range(num_channel_used):
                    if np.max(signal_downsampled[:, c]) == np.min(signal_downsampled[:, c]):
                        signal_normalized[:, c] = np.zeros(num_samples_used)
                    else:
                        signal_normalized[:, c] = zscore(signal_downsampled[:, c])
                feature[trial, repeat, char, :] = np.reshape(signal_normalized, [-1])
    return feature, labels


def load_dataset(datapath, subject):
    f = np.load(datapath+'processed/'+subject+'.npz')
    return f['featureTrain'], f['labelTrain']


if __name__ == '__main__':

    datapath = 'E:/eegdata/scutbci/p300speller/'

    import os
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    fs = 250
    f2 = 20
    order = 6
    fb, fa = signal.butter(order, 2 * f2 / fs, btype='low')
    # show_filter(fb, fa, fs)

    dfs = 6
    sampleseg = [0, int(0.6 * fs)]
    chanset = np.arange(30)

    subject = 'p300speller_yutianyou_20170313_1'
    file_train = datapath + subject + '.cnt'

    print('Load and extract continuous EEG into epochs for train data')
    dataTrain, targetTrain = load_eegdata(file_train, [fb, fa])
    print('Extract P300 features from epochs for train data')
    featureTrain, labelTrain = extract_feature(dataTrain, targetTrain, sampleseg, chanset, dfs)

    np.savez(datapath+'processed/'+subject+'.npz', featureTrain=featureTrain, labelTrain=labelTrain)


