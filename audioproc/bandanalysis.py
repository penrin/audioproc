import numpy as np
import math

'''
https://en.wikipedia.org/wiki/Octave_band
https://www.onosokki.co.jp/HP-WK/c_support/newreport/noise/souon_11.htm
http://kikakurui.com/c1/C1514-2002-01.html
JIS C 1514
'''


'''
center and limit frequencies of (1 / octN) octave band
'''
def bandfreq_base2(octN, fc_lim=(16, 20000), f_ref=1000):
    octave_ratio = 2
    if (octN % 2) == 1: # odd
        # decide n_start to fullfill f_lower[0] <= fc_lim[0]
        c_start = octN * np.log2(fc_lim[0] / f_ref) + 1 / 2
        n_start = int(np.floor(c_start))
        # decide n_start to fullfill f_upper[-1] >= fc_lim[1]
        c_end = octN * np.log2(fc_lim[1] / f_ref) - 1 / 2
        n_end = int(np.ceil(c_end))
        # f_center
        n = np.arange(n_start, n_end + 1)
        f_center = f_ref * octave_ratio ** (n / octN)
    else: # even
        # decide n_start to fullfill f_lower[0] <= fc_lim[0]
        c_start = octN * np.log2(fc_lim[0] / f_ref)
        n_start = int(np.floor(c_start))
        # decide n_start to fullfill f_upper[-1] >= fc_lim[1]
        c_end = octN * np.log2(fc_lim[1] / f_ref) - 1
        n_end = int(np.ceil(c_end))
        # f_center
        n = np.arange(n_start, n_end + 1)
        f_center = f_ref * octave_ratio ** ((2 * n + 1) / (2 * octN))
    # limit frequency
    f_lower = f_center * octave_ratio ** (-1 / (2 * octN))
    f_upper = f_center * octave_ratio ** (1 / (2 * octN))
    return f_center, f_lower, f_upper


def bandfreq_base10(octN, fc_lim=(16, 20000), f_ref=1000):
    octave_ratio = 10 ** 0.3 # 1.9953
    if (octN % 2) == 1: # odd
        # decide n_start to fullfill f_lower[0] <= fc_lim[0]
        c_start = octN * math.log(fc_lim[0] / f_ref, octave_ratio) + 1 / 2
        n_start = int(np.floor(c_start))
        # decide n_start to fullfill f_upper[-1] >= fc_lim[1]
        c_end = octN * math.log(fc_lim[1] / f_ref, octave_ratio) - 1 / 2
        n_end = int(np.ceil(c_end))
        # f_center
        n = np.arange(n_start, n_end + 1)
        f_center = f_ref * octave_ratio ** (n / octN)
    else: # even
        # decide n_start to fullfill f_lower[0] <= fc_lim[0]
        c_start = octN * math.log(fc_lim[0] / f_ref, octave_ratio)
        n_start = int(np.floor(c_start))
        # decide n_start to fullfill f_upper[-1] >= fc_lim[1]
        c_end = octN * math.log(fc_lim[1] / f_ref, octave_ratio) - 1
        n_end = int(np.ceil(c_end))
        # f_center
        n = np.arange(n_start, n_end + 1)
        f_center = f_ref * octave_ratio ** ((2 * n + 1) / (2 * octN))
    # limit frequency
    f_lower = f_center * octave_ratio ** (-1 / (2 * octN))
    f_upper = f_center * octave_ratio ** (1 / (2 * octN))
    return f_center, f_lower, f_upper 





import matplotlib.pyplot as plt


def ir2spectrum(ir, n, fs=48000):
    # n: FFT point
    freq = np.linspace(0, fs / 2, n // 2 + 1, endpoint=True)
    spectrum = np.abs(np.fft.rfft(ir, n=n)) ** 2
    return freq, spectrum


def ir2bandmean(ir, n, octN, fs=48e3, fc_lim=(16, 2e4), f_ref=1e3, base=10):
    if base == 10:
        fc, f1, f2 = bandfreq_base10(octN, fc_lim=fc_lim, f_ref=f_ref)
    elif base == 2:
        fc, f1, f2 = bandfreq_base2(octN, fc_lim=fc_lim, f_ref=f_ref)
    f, p = ir2spectrum(ir, n, fs=fs)
    E = np.empty(len(fc))
    for i in range(len(fc)):
        index = np.where((f1[i] <= f) & (f < f2[i]))[0]
        if len(index) < 10:
            print('fc %d Hz: very few bin %d' % (fc[i], len(index)))
        E[i] = np.mean(p[index])
    return fc, E


def ir2bandsum(ir, n, octN, fs=48e3, fc_lim=(16, 2e4), f_ref=1e3, base=10):
    if base == 10:
        fc, f1, f2 = bandfreq_base10(octN, fc_lim=fc_lim, f_ref=f_ref)
    elif base == 2:
        fc, f1, f2 = bandfreq_base2(octN, fc_lim=fc_lim, f_ref=f_ref)
    f, p = ir2spectrum(ir, n, fs=fs)
    E = np.empty(len(fc))
    for i in range(len(fc)):
        index = np.where((f1[i] <= f) & (f < f2[i]))[0]
        if len(index) < 10:
            print('very few bin')
        E[i] = np.sum(p[index])
    return fc, E


def plot_freqlevel(freq, level, xlim=(20, 2e4), ylim=None):
    plt.plot(freq, level, lw=1)
    plt.xscale('log')
    plt.grid(which='both')
    plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    # xlabel
    n1 = np.ceil(np.log10(xlim[0]))
    n2 = np.floor(np.log10(xlim[1]))
    xlabel = (10 ** np.arange(n1, n2 + 1)).astype(np.int)
    plt.xticks(xlabel, xlabel)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Level [dB]')
    return
    

def bin2bar(E, octN, fc_lim=(16, 20000), f_ref=1000, base=10):
    if base == 10:
        fc, f1, f2 = bandfreq_base10(octN, fc_lim=fc_lim, f_ref=f_ref)
    elif base == 2:
        fc, f1, f2 = bandfreq_base2(octN, fc_lim=fc_lim, f_ref=f_ref)
    f = np.empty(len(fc) * 2)
    f[0::2] = f1
    f[1::2] = f2
    y = np.empty(len(fc) * 2)
    y[0::2] = E
    y[1::2] = E
    return f, y

