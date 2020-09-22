# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal


def bandpower(x, fs, bands, method='fft'):
    """ Compute power in each frequency bin specified by bands from FFT result of
        x. By default, x is a real signal.
        Refer to https://github.com/forrestbao/pyeeg/blob/master/pyeeg/spectrum.py

        Parameters
        -----------
        x
            list
            a 1-D real time series.
        fs
            integer
            the sampling rate in physical frequency.
        bands
            list
            boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
            [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
            You can also use range() function of Python to generate equal bins and
            pass the generated list to this function.
            Each element of Band is a physical frequency and shall not exceed the
            Nyquist frequency, i.e., half of sampling frequency.

        method
            string
            power estimation method fft/burg/welch, default fft.

        Returns
        -------
        power
            list
            spectral power in each frequency bin.
        """

    L = len(x)
    if method == 'fft':
        pxx = np.fft.fft(x)
        pxx = abs(pxx)
        pxx = pxx[:L//2]
        f = np.arange(L / 2) / L
    elif method == 'welch':
        f, pxx = signal.welch(x, fs)
    elif method == 'periodogram':
        f, pxx = signal.periodogram(x, fs)
    else:
        assert 'unknown method'

    num_bands = len(bands) - 1
    pxx_bands = np.zeros(num_bands)
    for i in range(0, num_bands):
        f1 = float(bands[i]) / fs
        f2 = float(bands[i + 1]) / fs
        indices = np.argwhere((f >= f1) & (f < f2))
        pxx_bands[i] = sum(pxx[indices])
    return pxx_bands


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fs = 1000
    T = 1 / fs
    L = 1500
    t = np.arange(L) * T
    f1 = np.arange(L / 2) / L
    nfft = int(2 ** np.ceil(np.log2(L)))
    s = 0.7 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    x = s + 2 * np.random.randn(t.size)

    y = np.fft.fft(x)
    pxx1 = abs(y)
    pxx1 = pxx1[:L//2]
    f2, pxx2 = signal.periodogram(x)
    pxx2[0] = np.median(pxx2[:3])
    f3, pxx3 = signal.welch(x)

    plt.subplot(411)
    plt.plot(t * fs, x)
    plt.title('Signal Corrupted with Zero-Mean Random Noise')
    plt.xlabel('t (milliseconds)')
    plt.ylabel('X(t)')

    plt.subplot(412)
    plt.plot(f1 * fs, 10 * np.log10(pxx1))
    plt.title('Single-Sided Amplitude Spectrum of X(t)')
    plt.xlabel('f (Hz)')
    plt.ylabel('10*log10|P1(f)|')

    plt.subplot(413)
    plt.plot(f2, 10 * np.log10(pxx2))
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power (dB)')
    plt.title('Periodogram')

    plt.subplot(414)
    plt.plot(f3, 10 * np.log10(pxx3))
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power (dB)')
    plt.title('Welchs')

    plt.show()