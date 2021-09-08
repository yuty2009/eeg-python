# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset
from common.datawrapper import read_matdata, read_gdfdata
from common.signalproc import *


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


_available_dataset = [
    'bcicomp2005IVa',     # 2 class (L, R)
    'bcicomp2005IIIa',    # 4 class (L, R, F, T)
    'bcicomp2008IIa',     # 4 class (L, R, F, T)
    ]

def load_eegdata(setname, datapath, subject):
    assert setname in _available_dataset, 'Unknown dataset name ' + setname
    if setname == 'bcicomp2005IVa':
        filepath = datapath+'data_set_IVa_'+subject+'.mat'
        labelpath = datapath+'true_labels_'+subject+'.mat'
        dataTrain, targetTrain, dataTest, targetTest = \
            load_eegdata_bcicomp2005IVa(filepath, labelpath)
    elif setname in ['bcicomp2005IIIa', 'bcicomp2005IIIa_2c']:
        filepath = datapath+subject+'b.mat'
        labelpath = datapath+'true_label_'+subject+'.mat'
        dataTrain, targetTrain, dataTest, targetTest = \
            load_eegdata_bcicomp2005IIIa(filepath, labelpath)
    elif setname in ['bcicomp2008IIa', 'bcicomp2008IIa_2c']:
        fdatatrain = datapath+subject+'T.gdf'
        flabeltrain = datapath+subject+'T.mat'
        fdatatest = datapath+subject+'E.gdf'
        flabeltest = datapath+subject+'E.mat'
        dataTrain, targetTrain, dataTest, targetTest = \
            load_eegdata_bcicomp2008IIa(fdatatrain, flabeltrain, fdatatest, flabeltest)
    return dataTrain, targetTrain, dataTest, targetTest


def load_eegdata_bcicomp2005IVa(filepath, labelpath):
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
    timewin = [-0.5, 4.0] # 4.5s data including 0.5s pre-task
    sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
    num_samples = sampleseg[1] - sampleseg[0]
    num_channels = cnt.shape[1]

    targetTrain = np.zeros(num_train, dtype=np.int)
    dataTrain = np.zeros([num_train, num_samples, num_channels])
    for i in range(num_train):
        begin = pos[i]
        end = begin + num_samples
        dataTrain[i,:,:] = cnt[begin:end,:]
        targetTrain[i] = code[i]

    targetTest = np.zeros(num_test, dtype=np.int)
    dataTest = np.zeros([num_test, num_samples, num_channels])
    for i in range(num_test):
        begin = pos[num_train+i]
        end = begin + num_samples
        dataTest[i,:,:] = cnt[begin:end,:]
        targetTest[i] = label[num_train+i]

    return dataTrain, targetTrain, dataTest, targetTest


def load_eegdata_bcicomp2005IIIa(filepath, labelpath):
    label_key = os.path.basename(labelpath)
    label_key = label_key[:-4]
    labeldata = read_matdata(labelpath, [label_key])
    label = labeldata[label_key]

    data = read_matdata(filepath, ['s', 'HDR'])
    s = data['s']
    fs = data['HDR']['SampleRate'][0][0][0][0]
    pos = data['HDR']['TRIG'][0][0].squeeze()
    code = data['HDR']['Classlabel'][0][0].squeeze()
    index_train = np.argwhere(~np.isnan(code))
    index_test = np.argwhere(np.isnan(code))

    num_train = len(index_train)
    num_test = len(index_test)
    timewin = [-0.5, 4.0] # 4.5s data including 0.5s pre-task
    sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
    num_samples = sampleseg[1] - sampleseg[0]
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
        ii = index_train[i, 0]
        begin = pos[ii] + sampleseg[0] + 3*fs # 3 seconds prepare
        end = begin + num_samples
        dataTrain[i,:,:] = s[begin:end,:]
        targetTrain[i] = code[ii]

    targetTest = np.zeros(num_test, dtype=np.int)
    dataTest = np.zeros([num_test, num_samples, num_channels])
    for i in range(num_test):
        ii = index_test[i, 0]
        begin = pos[ii] + sampleseg[0] + 3*fs # 3 seconds prepare
        end = begin + num_samples
        dataTest[i,:,:] = s[begin:end,:]
    targetTest = np.squeeze(label[index_test]).astype(np.int)

    return dataTrain, targetTrain, dataTest, targetTest


def load_eegdata_bcicomp2008IIa(fdatatrain, flabeltrain, fdatatest, flabeltest):
    startcode = 768 # trial start event code
    s, events, clabs = read_gdfdata(fdatatrain)
    s = 1e6 * s # convert to millvolt for numerical stability
    pos = events['pos']
    code = events['type']
    indices = np.argwhere(code == startcode) # 768

    num_train = len(indices)
    fs = 250
    timewin = [-0.5, 4.0] # 4.5s data including 0.5s pre-task
    sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
    num_samples = sampleseg[1] - sampleseg[0]
    num_channels = s.shape[1]
    dataTrain = np.zeros([num_train, num_samples, num_channels])
    for i in range(num_train):
        begin = pos[indices[i, 0]] + sampleseg[0] + 2*fs # 2 seconds prepare
        end = begin + num_samples
        dataTrain[i,:,:] = s[begin:end,:]
    labeldata = read_matdata(flabeltrain, ['classlabel'])
    targetTrain = np.array(np.squeeze(labeldata['classlabel']), dtype=np.int)

    s, events, clabs = read_gdfdata(fdatatest)
    s = 1e6 * s # convert to millvolt for numerical stability
    pos = events['pos']
    code = events['type']
    indices = np.argwhere(code == startcode) # 768

    num_test = len(indices)
    dataTest = np.zeros([num_test, num_samples, num_channels])
    for i in range(num_test):
        begin = pos[indices[i, 0]] + sampleseg[0] + 2*fs # 2 seconds prepare
        end = begin + num_samples
        dataTest[i,:,:] = s[begin:end,:]
    labeldata = read_matdata(flabeltest, ['classlabel'])
    targetTest = np.array(np.squeeze(labeldata['classlabel']), dtype=np.int)
    
    return dataTrain, targetTrain, dataTest, targetTest


def extract_rawfeature(data, target, sampleseg, chanset, filter=None, standardize=False):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_samples_used = sample_end - sample_begin
    num_channel_used = len(chanset)

    # show_filtering_result(filter[0], filter[1], data[0,:,0])

    labels = target
    features = np.zeros([num_trials, num_samples_used, num_channel_used])
    for i in range(num_trials):
        signal_filtered = data[i]
        if filter is not None:
            signal_filtered = signal.lfilter(filter[0], filter[1], signal_filtered, axis=0)
            # signal_filtered = signal.filtfilt(filter[0], filter[1], signal_epoch, axis=0)
        if standardize:
            # init_block_size = 1000 this param setting has a big impact on the result
            signal_filtered = exponential_running_standardize(signal_filtered, init_block_size=1000)
        features[i] = signal_filtered[sample_begin:sample_end, chanset]

    return features, labels


def extract_variance(data, target, sampleseg, chanset, filter=None):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_channel_used = len(chanset)

    # show_filtering_result(filter[0], filter[1], data[0,:,0])

    labels = target
    Rs = np.zeros([num_trials, num_channel_used, num_channel_used])
    for i in range(num_trials):
        signal_filtered = data[i]
        if filter is not None:
            signal_filtered = signal.lfilter(filter[0], filter[1], signal_filtered, axis=0)
            # signal_filtered = signal.filtfilt(filter[0], filter[1], signal_epoch, axis=0)
        signal_filtered = signal_filtered[sample_begin:sample_end, chanset]
        cov_tmp = np.dot(signal_filtered.T, signal_filtered)
        Rs[i] = cov_tmp/np.trace(cov_tmp)

    return Rs, labels


def extract_variance_multiband(data, target, bands, sampleseg, chanset):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_channel_used = len(chanset)

    fs = 100
    order = 3
    num_bands = len(bands)
    Rss = []
    for k in range(num_bands):
        f1, f2 = bands[k]
        fb, fa = signal.butter(order, [2*f1/fs, 2*f2/fs], btype='bandpass')
        # fb, fa = signal.cheby2(order, 40, [2*f1/fs, 2*f2/fs], btype='bandpass')
        Rs = np.zeros([num_trials, num_channel_used, num_channel_used])
        for i in range(num_trials):
            signal_filtered = data[i]
            signal_filtered = signal.lfilter(filter[0], filter[1], signal_filtered, axis=0)
            # signal_filtered = signal.filtfilt(filter[0], filter[1], signal_epoch, axis=0)
            signal_filtered = signal_filtered[sample_begin:sample_end, chanset]
            cov_tmp = np.dot(signal_filtered.T, signal_filtered)
            Rs[i] = cov_tmp/np.trace(cov_tmp)
        Rss.append(Rs)

    labels = target

    return Rss, labels


def load_dataset_preprocessed(datapath, subject):
    f = np.load(datapath+'processed/'+subject+'.npz')
    return f['dataTrain'], f['targetTrain'], f['dataTest'], f['targetTest']


if __name__ == '__main__':

    # setname = 'bcicomp2005IVa'
    # datapath = 'E:/bcicompetition/bci2005/IVa/'
    # subjects = ['aa', 'al', 'av', 'aw', 'ay']
    setname = 'bcicomp2008IIa'
    datapath = 'E:/bcicompetition/bci2008/IIa/'
    subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

    import os
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    for ss in range(len(subjects)):
        subject = subjects[ss]

        print('Load and extract continuous EEG into epochs for subject '+subject)
        dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)

        np.savez(datapath+'processed/'+subject+'.npz',
                 dataTrain=dataTrain, targetTrain=targetTrain, dataTest=dataTest, targetTest=targetTest)


