# coding: utf-8
import numpy as np
from scipy.signal import get_window
from scipy.signal import fftconvolve


# generate swept-sine signal
def gentsp(n=18):
    N = 2 ** n
    m = N // 4
    SS = np.r_[
        np.exp(-1.j * np.pi / N * np.arange(0, N // 2 + 1) ** 2),
        np.exp(1.j * np.pi / N * (N - np.arange(N // 2 + 1, N)) ** 2)
        ]
    upward = np.roll(np.fft.ifft(SS).real, m)
    upward /= np.max(np.abs(upward))
    downward = upward[::-1].copy()
    return upward, downward


# Exponent of next higher power of 2
def nextpow2(n):
    l = np.ceil(np.log2(n))
    m = int(np.log2(2 ** l))
    return m


# Cross-correlation function
def fftxcorr(x, y):
    # Rxy = E[x(n) y(n + m)]
    fftpoint = 2 ** nextpow2(len(x) + len(y) - 1)
    X = np.fft.rfft(x, n=fftpoint)
    Y = np.fft.rfft(y, n=fftpoint)
    c = np.fft.irfft(np.conj(X) * Y)
    return c


# Auto-correlation function
def fftacorr(x):
    fftpoint = 2 ** nextpow2(2 * len(x) - 1)
    X = np.fft.rfft(x, n=fftpoint)
    c = np.fft.irfft(np.conj(X) * X)
    return c


# Cross-correlation function (returned with lag time)
def xcorr(x, y):
    fftpoint = 2 ** nextpow2(len(x) + len(y) - 1)
    X = np.fft.rfft(x[::-1], n=fftpoint)
    Y = np.fft.rfft(y, n=fftpoint)
    cf = np.fft.irfft(X * Y)[:len(x) + len(y) - 1]
    n_axis = np.arange(len(x) + len(y) - 1) - len(x) + 1
    return n_axis, cf


# L times upsample for fixed sample rate signal
def upsample(x, K, N, window='hamming'):
    '''
    Parameters
    ----------
    K: int
        The multiple of upsampling.
    N: int
        The number of taps of interpolation function.
        The longer this number, the higher the accuracy,
        but the higher the calculation load.
    window : string or tuple of string and parameter values
        Desired window to use. See `scipy.signal.get_window` for a list
        of windows and required parameters.
    '''
    
    if type(K) != int:
        print('Only integer multiples please.')
    
    # upsample
    x_upsamp = np.zeros((x.shape[0] - 1) * K + 1)
    x_upsamp[::K] = x[:]

    # LPF
    n = np.arange(N) - (N - 1) / 2
    w = get_window(window, N)
    LPF = w * np.sinc(n / K)
    y = fftconvolve(x_upsamp, LPF)
    return y

