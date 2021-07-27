# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as signal
from common.datawrapper import *
from common.signalproc import *


def load_dataset(filepath, labelpath):
    data = read_matdata(filepath, ['clab', 'x_train', 'y_train', 'x_test'])
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    with open(labelpath, 'r') as myfile:
        y_test = myfile.readlines()
        y_test = [int(x) for x in y_test]
        y_test = np.array(y_test)
    return x_train, y_train.squeeze(), x_test, y_test.squeeze()


def extract_bandpower(data, bands, fs):
    num_samples, num_channels, num_trials = data.shape
    num_bands = len(bands) - 1
    
    features = np.zeros((num_trials, num_bands, num_channels))
    for i in range(num_trials):
        for j in range(num_channels):
            data_1 = data[:,j,i]
            f, pxx = signal.welch(data_1, fs)
            pxx_bands = np.zeros(num_bands)
            for k in range(0, num_bands):
                f1 = bands[k]
                f2 = bands[k + 1]
                indices = np.argwhere((f >= f1) & (f < f2))
                pxx_bands[k] = sum(pxx[indices])

            features[i, :, j] = pxx_bands
    
    return np.reshape(features, [num_trials, -1])


if __name__ == '__main__':

    datapath = 'E:/bcicompetition/bci2003/IV/sp1s_aa_1000Hz.mat'
    labelpath = 'E:/bcicompetition/bci2003/IV/labels_data_set_iv.txt'
    x_train, y_train, x_test, y_test = load_dataset(datapath, labelpath)
    print('Data loaded with %d training samples and %d testing samples' % (y_train.size, y_test.size))

    fs = 1000
    num_samples, num_channels, num_trials = x_train.shape
    nfft = num_samples
    overlap = 64
    window = signal.windows.hann(nfft, True)
    data_1 = x_train[:,0,0]
    f, pxx = signal.welch(data_1, fs, window=window, noverlap=overlap, nfft=nfft)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(f, np.sqrt(pxx))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.show()
