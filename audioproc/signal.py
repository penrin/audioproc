# coding: utf-8
import numpy as np


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
    fftsize = 2 ** nextpow2(len(x) + len(y) - 1)
    x_zeropad = np.zeros(fftsize)
    y_zeropad = np.zeros(fftsize)
    x_zeropad[:x.shape[0]] = x
    y_zeropad[:y.shape[0]] = y
    X = np.fft.rfft(x_zeropad)
    Y = np.fft.rfft(y_zeropad)
    c = np.fft.irfft(np.conj(X) * Y)
    return c


# Auto-correlation function
def fftacorr(x):
    fftsize = 2 ** nextpow2(2 * len(x) - 1)
    x_zeropad = np.zeros(fftsize)
    x_zeropad[:x.shape[0]] = x
    X = np.fft.rfft(x_zeropad)
    c = np.fft.irfft(np.conj(X) * X)
    return c


