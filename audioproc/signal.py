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
    
