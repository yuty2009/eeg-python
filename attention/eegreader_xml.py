# -*- coding: utf-8 -*-

import numpy as np
from common.datawrapper import *
from common.temporalfilter import *
from scipy.stats import zscore


def load_eegdata(filepath, filter=None, dfs=None):
    eegdata, events = read_xmldata(filepath)
    return extract_eegdata(eegdata, events, filter, dfs)


def read_xmldata(filepath):
    from xml.etree.ElementTree import parse
    doc = parse(filepath)
    root = doc.getroot()
    data = [int(x) for x in root.findtext('data').strip().split(' ')]
    event = [int(x) for x in root.findtext('event').strip().split(' ')]
    return np.array(data), np.array(event)


def extract_eegdata(eegdata, events, filter=None, dfs=None):
    eegdata = eegdata / 100.0
    fs = 250
    if filter is not None:
        fb, fa = filter
        # show_filtering_result(fb, fa, eegdata[:1000])
        # eegdata = signal.filtfilt(fb, fa, eegdata)
    dfs = 1 if dfs is None else dfs

    eventpos = np.logical_and(events > 0, events <= 50).nonzero()[0]
    eventtype = events[eventpos]
    num_trials = len(eventpos)

    offset = 3
    epoch_lens = [60*fs, 30*fs]
    start = offset * fs
    ends = [start + epoch_lens[0], start + epoch_lens[1]]
    window_len = 10 * fs
    sample_len = int(window_len / dfs)
    window_num = [int(epoch_lens[0] / window_len), int(epoch_lens[1] / window_len)]

    data = []
    target = []
    for i in range(num_trials):
        if eventtype[i] <= 30:
            signal_i = eegdata[eventpos[i]+start:eventpos[i]+ends[0]]
            event_i = int(eventtype[i] % 3) # 0: 安静 1：注意  2：放松
            target_i = event_i * np.ones(window_num[0])
        else:
            signal_i = eegdata[eventpos[i]+start:eventpos[i]+ends[1]]
            event_i = 3 # 3: 闭眼
            target_i = event_i * np.ones(window_num[1])
        if dfs is not 1:
            signal_i = signal.decimate(signal_i, dfs)
        data_i = np.reshape(signal_i, [sample_len, -1])
        data.append(data_i.T)
        target.append(target_i)
    data = np.concatenate(data, axis=0)
    target = np.concatenate(target, axis=0).astype(int)

    return data, target


def extract_feature_hilbert(data, target):
    fs = 125
    num_examples = len(target)
    num_features = 10 * fs
    num_features_downsampled = 100

    # filter bank design
    order = 4
    band_width = 1.7  # 2
    low_bands = np.arange(1, 1.7 * 35, 1.7)  # (1, 70, 2)
    high_bands = low_bands + band_width
    FB = []
    FA = []
    for lb, hb in zip(low_bands, high_bands):
        b, a = signal.butter(order, (lb / (0.5 * fs), hb / (0.5 * fs)), btype='bandpass', analog=False)
        FB.append(b)
        FA.append(a)

    # design 0.1Hz high-pass filter and 50hz notch filter
    b1, a1 = signal.butter(order, 0.1 / (0.5 * fs), btype='highpass')
    Q = 30.0  # Quality factor
    b2, a2 = signal.iirnotch(50 / (0.5 * fs), Q)

    # feature extraction
    features = np.zeros((num_examples, len(FA), num_features))
    features_downsampled = np.zeros((num_examples, len(FA), num_features_downsampled))
    for i in range(num_examples):
        data_i = data[i, :]
        # 0.1Hz high-pass filter & 50Hz notch filter
        data_i = data_i - np.mean(data_i)
        data_i = signal.filtfilt(b1, a1, data_i)
        data_i = signal.filtfilt(b2, a2, data_i)

        feature_i = np.zeros((len(FA), num_features))
        j = 0
        for fb, fa in zip(FB, FA):
            fsig = signal.filtfilt(fb, fa, data_i)
            hfsig = signal.hilbert(fsig)
            feature_i[j] = np.abs(hfsig)
            j += 1
        features[i, :, :] = feature_i  # feature: 1*35*1250
        features_downsampled[i, :, :] = signal.resample(feature_i, num_features_downsampled, axis=1)

    return features_downsampled, target


def load_dataset(datapath):
    f = np.load(datapath+'processed/data_hilbert.npz', allow_pickle=True)
    return f['data']


if __name__ == '__main__':

    # datapath = 'e:/eegdata/attention/Headband/data4class/'
    datapath = '/home/yuty2009/data/eegdata/attention/Headband/data4class/'

    import os
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    fs = 250
    f1 = 0.1
    order = 4
    fb, fa = signal.butter(order, 2 * f1 / fs, btype='high')
    # show_filter(fb, fa, fs)

    import glob
    data_all = []
    for filepath in glob.glob(datapath + "/*.xml"):
        print('Load and extract feature for %s' % filepath)
        data, target = load_eegdata(filepath, filter=[fb, fa], dfs=2)
        features, labels = extract_feature_hilbert(data, target)
        data_i = dict()
        data_i['features'] = features
        data_i['labels'] = labels
        data_all.append(data_i)
    np.savez(datapath + 'processed/data_hilbert.npz',data=data_all)

