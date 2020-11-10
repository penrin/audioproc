# coding: utf-8
import numpy as np
from scipy.signal import get_window
from scipy.signal import fftconvolve
import audioproc as ap


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
    return int(np.ceil(np.log2(n)))


# The maximum absolute value
def absmax(x, axis=None):
    return np.max(np.abs(x), axis=axis)


# The argument of the maximum absolute value
def argabsmax(x, axis=None):
    return np.argmax(np.abs(x), axis=axis)


# The minimum absolute value
def absmin(x, axis=None):
    return np.min(np.abs(x), axis=axis)


# The argument of the minimum absolute value
def argabsmin(x, axis=None):
    return np.argmin(np.abs(x), axis=axis)


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


# amplitude limiter
def limiter(signal, threshold, deepcopy=True):
    if deepcopy:
        s = np.copy(signal)
    else:
        s = signal
    index = np.where(np.abs(s) > np.abs(threshold))
    s[index] = np.sign(s[index]) * np.abs(threshold)
    return s


# convolution using overlap-save method
# with low memory consumption for long input
def conv_lessmemory(longinput, fir, fftpoint, verbose=False):
    len_input = longinput.shape[-1]
    M = fir.shape[-1]
    N = fftpoint
    L = N - (M - 1)

    if longinput.ndim != 2:
        raise Exception('longinput must be 2 dim')
    if fir.ndim != 3:
        raise Exception('fir must be 3 dim')
    if longinput.shape[0] != fir.shape[1]:
        raise Exception('fir shape does not match input')

    n_input = longinput.shape[0]
    n_output = fir.shape[0]

    if L < 1:
        raise Exception('fftpoint must be more than %d' % M)

    fir_f = np.fft.rfft(fir, n=N)
    del fir
    
    block_in = np.empty([n_input, 1, N])
    block_in[:, 0, L:] = 0.
    
    point_read = 0

    len_out = len_input + M - 1
    out = np.empty([n_output, len_out])
    out_cnt = 0
    nblocks = int(np.ceil(len_out / L))
    
    if verbose:
        print('fftpoint:%d, ' % N, end='')
        print('blocksize:%d, ' % L, end='')
        print('nblocks:%d' % nblocks)
        pg = ap.ProgressBar2(nblocks, slug='=', space=' ')

    for l in range(nblocks):
        
        # overlap
        block_in[:, 0, :-L] = block_in[:, 0, L:]
        
        # read input
        if (len_input - point_read) >= L:
            block_in[:, 0, -L:] = longinput[:, point_read:point_read+L]
            point_read += L
        else:
            ll = len_input - point_read
            block_in[:, 0, -L:-L+ll] = longinput[:, point_read:]
            block_in[:, 0, -L+ll:] = 0
            point_read += ll
        
        # convolution
        block_in_f = np.fft.rfft(block_in, n=N)
        block_out_f = np.matmul(
                fir_f.transpose(2, 0, 1), block_in_f.transpose(2, 0, 1)
                ).transpose(1, 2, 0)
        block_out = np.fft.irfft(block_out_f)[:, 0, -L:]
        
        # write output
        if (len_out - out_cnt) >= L:
            out[:, out_cnt:out_cnt+L] = block_out
            out_cnt += L
        else:
            out[:, out_cnt:] = block_out[:, :len_out - out_cnt]
            out_cnt = len_out

        if verbose:
            pg.bar()
    
    return out



# convolution using overlap-save method
# with low memory consumption for long input (Frequency-domain FIR version)
#
# longinput: time-domain
# rfft_fir : freq-domain (by np.fft.rfft that fftpoint have to be even number)
# ntaps_fir: length of FIR in time-domain
#
def conv_lessmemory_fdomfir(longinput, rfft_fir, ntaps_fir, verbose=False):
    len_input = longinput.shape[-1]
    M = ntaps_fir
    N = 2 * rfft_fir.shape[-1] - 2 # original N is an even number
    L = N - (M - 1)

    if longinput.ndim != 2:
        raise Exception('longinput must be 2 dim')
    if rfft_fir.ndim != 3:
        raise Exception('fir must be 3 dim')
    if longinput.shape[0] != rfft_fir.shape[1]:
        raise Exception('fir shape does not match input')

    n_input = longinput.shape[0]
    n_output = rfft_fir.shape[0]

    if L < 1:
        raise Exception('fftpoint must be more than %d' % M)

    fir_f = rfft_fir
    
    block_in = np.empty([n_input, 1, N])
    block_in[:, 0, L:] = 0.
    
    point_read = 0

    len_out = len_input + M - 1
    out = np.empty([n_output, len_out])
    out_cnt = 0
    nblocks = int(np.ceil(len_out / L))
    
    if verbose:
        print('fftpoint:%d, ' % N, end='')
        print('blocksize:%d, ' % L, end='')
        print('nblocks:%d' % nblocks)
        pg = ap.ProgressBar2(nblocks, slug='=', space=' ')

    for l in range(nblocks):
        
        # overlap
        block_in[:, 0, :-L] = block_in[:, 0, L:]
        
        # read input
        if (len_input - point_read) >= L:
            block_in[:, 0, -L:] = longinput[:, point_read:point_read+L]
            point_read += L
        else:
            ll = len_input - point_read
            block_in[:, 0, -L:-L+ll] = longinput[:, point_read:]
            block_in[:, 0, -L+ll:] = 0
            point_read += ll
        
        # convolution
        block_in_f = np.fft.rfft(block_in, n=N)
        block_out_f = np.matmul(
                fir_f.transpose(2, 0, 1), block_in_f.transpose(2, 0, 1)
                ).transpose(1, 2, 0)
        block_out = np.fft.irfft(block_out_f)[:, 0, -L:]
        
        # write output
        if (len_out - out_cnt) >= L:
            out[:, out_cnt:out_cnt+L] = block_out
            out_cnt += L
        else:
            out[:, out_cnt:] = block_out[:, :len_out - out_cnt]
            out_cnt = len_out

        if verbose:
            pg.bar()
    
    return out



# combine two MIMO FIR
# shape of fir1 and fir2 -> (out-ch, in-ch, taps)
# return fir2 @ fir1
def combine_fir(fir1, fir2):
    len_new = fir1.shape[-1] + fir2.shape[-1] - 1
    fftpt = 2 ** ap.nextpow2(len_new)
    fir1_f = np.fft.rfft(fir1, n=fftpt)
    fir2_f = np.fft.rfft(fir2, n=fftpt)
    new_f = np.matmul(
            fir2_f.transpose(2, 0, 1), fir1_f.transpose(2, 0, 1)
            ).transpose(1, 2, 0)
    new = np.fft.irfft(new_f)[:, :, :len_new]
    return new


