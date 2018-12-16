# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import butter, lfilter, filtfilt, decimate


def show_filter(b, a, fs=None):
    from scipy.signal import freqz
    import matplotlib.pyplot as plt
    w, h = freqz(b, a)
    if fs == None:
        plt.semilogx(w, 20 * np.log10(abs(h)))
    else:
        plt.plot((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude [dB]')
    plt.grid(True)
    plt.show()


def show_filtering_result(b, a, x):
    import matplotlib.pyplot as plt
    x1 = lfilter(b, a, x)
    x2 = filtfilt(b, a, x)
    plt.plot(x, 'k-', label='input')
    plt.plot(x1, 'b-', label='lfilter')
    plt.plot(x2, 'c-', label='filtfilt')
    plt.legend(loc='best')
    plt.show()
